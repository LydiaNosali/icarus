"""Implementations of all on-path strategies"""
import ast
from collections import defaultdict
import csv
import logging
import os
import pickle
import random
import time
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import networkx as nx
from joblib import Parallel, delayed


from icarus.registry import register_strategy
from icarus.util import inheritdoc, path_links

from .base import Strategy

__all__ = [
    "Partition",
    "Edge",
    "LeaveCopyEverywhere",
    "LeaveCopyDown",
    "ProbCache",
    "LRUCost",
    "CacheLessForMore",
    "RandomBernoulli",
    "RandomChoice",
    "Cost",
]

logger = logging.getLogger("main")

@register_strategy("PARTITION")
class Partition(Strategy):
    """Partition caching strategy.

    In this strategy the network is divided into as many partitions as the number
    of caching nodes and each receiver is statically mapped to one and only one
    caching node. When a request is issued it is forwarded to the cache mapped
    to the receiver. In case of a miss the request is routed to the source and
    then returned to cache, which will store it and forward it back to the
    receiver.

    This requires median cache placement, which optimizes the placement of
    caches for this strategy.

    This strategy is normally used with a small number of caching nodes. This
    is the the behaviour normally adopted by Network CDN (NCDN). Google Global
    Cache (GGC) operates this way.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller):
        super().__init__(view, controller)
        if "cache_assignment" not in self.view.topology().graph:
            raise ValueError(
                "The topology does not have cache assignment "
                "information. Have you used the optimal median "
                "cache assignment?"
            )
        self.cache_assignment = self.view.topology().graph["cache_assignment"]

    @inheritdoc(Strategy)
    def  process_event(self, time, receiver, content, size, priority, log):
        source = self.view.content_source(content)
        self.controller.start_session(time, receiver, content, log, priority)
        cache = self.cache_assignment[receiver]
        self.controller.forward_request_path(receiver, cache)
        if not self.controller.get_content(cache):
            self.controller.forward_request_path(cache, source)
            self.controller.get_content(source)
            self.controller.forward_content_path(source, cache)
            self.controller.put_content(cache)
        self.controller.forward_content_path(cache, receiver)
        self.controller.end_session()


@register_strategy("EDGE")
class Edge(Strategy):
    """Edge caching strategy.

    In this strategy only a cache at the edge is looked up before forwarding
    a content request to the original source.

    In practice, this is like an LCE but it only queries the first cache it
    finds in the path. It is assumed to be used with a topology where each
    PoP has a cache but it simulates a case where the cache is actually further
    down the access network and it is not looked up for transit traffic passing
    through the PoP but only for PoP-originated requests.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller):
        super().__init__(view, controller)

    @inheritdoc(Strategy)
    def  process_event(self, time, receiver, content, size, priority, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log, priority)
        edge_cache = None
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                edge_cache = v
                if self.controller.get_content(v, size=size, priority=priority):
                    serving_node = v
                else:
                    # Cache miss, get content from source
                    self.controller.forward_request_path(v, source)
                    self.controller.get_content(source)
                    serving_node = source
                break
        else:
            # No caches on the path at all, get it from source
            self.controller.get_content(v, size=size, priority=priority)
            serving_node = v

        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        self.controller.forward_content_path(serving_node, receiver, path=path)
        if serving_node == source:
            self.controller.put_content(edge_cache)
        self.controller.end_session()


@register_strategy("LCE")
class LeaveCopyEverywhere(Strategy):
    """Leave Copy Everywhere (LCE) strategy.

    In this strategy a copy of a content is replicated at any cache on the
    path between serving node and receiver.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super().__init__(view, controller)

    @inheritdoc(Strategy)
    def  process_event(self, time, receiver, content, size, priority, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log, priority)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v, size=size, priority=priority):
                    serving_node = v
                    break
        
        else:
            # No cache hits, get content from source
            self.controller.get_content(v, size=size, priority=priority)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v, size=size, priority=priority)
            if self.view.has_cache(v):
                # insert content
                self.controller.put_content(v, size=size, priority=priority)
        self.controller.end_session()


@register_strategy("LCD")
class LeaveCopyDown(Strategy):
    """Leave Copy Down (LCD) strategy.

    According to this strategy, one copy of a content is replicated only in
    the caching node you hop away from the serving node in the direction of
    the receiver. This strategy is described in [2]_.

    Rereferences
    ------------
    ..[1] N. Laoutaris, H. Che, i. Stavrakakis, The LCD interconnection of LRU
          caches and its analysis.
          Available: http://cs-people.bu.edu/nlaout/analysis_PEVA.pdf
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super().__init__(view, controller)

    @inheritdoc(Strategy)
    def  process_event(self, time, receiver, content, size, priority, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log, priority)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v, size=size, priority=priority):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v, size=size, priority=priority)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        # Leave a copy of the content only in the cache one level down the hit
        # caching node
        copied = False
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v, size=size, priority=priority)
            if not copied and v != receiver and self.view.has_cache(v):
                self.controller.put_content(v, size=size, priority=priority)
                copied = True
        self.controller.end_session()


@register_strategy("PROB_CACHE")
class ProbCache(Strategy):
    """ProbCache strategy [3]_

    This strategy caches content objects probabilistically on a path with a
    probability depending on various factors, including distance from source
    and destination and caching space available on the path.

    This strategy was originally proposed in [2]_ and extended in [3]_. This
    class implements the extended version described in [3]_. In the extended
    version of ProbCache the :math`x/c` factor of the ProbCache equation is
    raised to the power of :math`c`.

    References
    ----------
    ..[2] I. Psaras, W. Chai, G. Pavlou, Probabilistic In-Network Caching for
          Information-Centric Networks, in Proc. of ACM SIGCOMM ICN '12
          Available: http://www.ee.ucl.ac.uk/~uceeips/prob-cache-icn-sigcomm12.pdf
    ..[3] I. Psaras, W. Chai, G. Pavlou, In-Network Cache Management and
          Resource Allocation for Information-Centric Networks, IEEE
          Transactions on Parallel and Distributed Systems, 22 May 2014
          Available: http://doi.ieeecomputersociety.org/10.1109/TPDS.2013.304
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, t_tw=10):
        super().__init__(view, controller)
        self.t_tw = t_tw
        self.cache_size = view.cache_nodes(size=True)

    @inheritdoc(Strategy)
    def  process_event(self, time, receiver, content, size, priority, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log, priority)
        for hop in range(1, len(path)):
            u = path[hop - 1]
            v = path[hop]
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v, size=size, priority=priority):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v, size=size, priority=priority)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        c = len([node for node in path if self.view.has_cache(node)])
        x = 0.0
        for hop in range(1, len(path)):
            u = path[hop - 1]
            v = path[hop]
            N = sum(
                [self.cache_size[n] for n in path[hop - 1 :] if n in self.cache_size]
            )
            if v in self.cache_size:
                x += 1
            self.controller.forward_content_hop(u, v, size=size, priority=priority)
            if v != receiver and v in self.cache_size:
                # The (x/c) factor raised to the power of "c" according to the
                # extended version of ProbCache published in IEEE TPDS
                prob_cache = float(N) / (self.t_tw * self.cache_size[v]) * (x / c) ** c
                if random.random() < prob_cache:
                    self.controller.put_content(v, size=size, priority=priority)
        self.controller.end_session()


@register_strategy("CL4M")
class CacheLessForMore(Strategy):
    """Cache less for more strategy [4]_.

    This strategy caches items only once in the delivery path, precisely in the
    node with the greatest betweenness centrality (i.e., that is traversed by
    the greatest number of shortest paths). If the argument *use_ego_betw* is
    set to *True* then the betweenness centrality of the ego-network is used
    instead.

    References
    ----------
    ..[4] W. Chai, D. He, I. Psaras, G. Pavlou, Cache Less for More in
          Information-centric Networks, in IFIP NETWORKING '12
          Available: http://www.ee.ucl.ac.uk/~uceeips/centrality-networking12.pdf
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, use_ego_betw=False, **kwargs):
        super().__init__(view, controller)
        topology = view.topology()
        if use_ego_betw:
            self.betw = {
                v: nx.betweenness_centrality(nx.ego_graph(topology, v))[v]
                for v in topology.nodes()
            }
        else:
            self.betw = nx.betweenness_centrality(topology)

    @inheritdoc(Strategy)
    def  process_event(self, time, receiver, content, size, priority, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log, priority)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v, size=size, priority=priority):
                    serving_node = v
                    break
        # No cache hits, get content from source
        else:
            self.controller.get_content(v, size=size, priority=priority)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        # get the cache with maximum betweenness centrality
        # if there are more than one cache with max betw then pick the one
        # closer to the receiver
        max_betw = -1
        designated_cache = None
        for v in path[1:]:
            if self.view.has_cache(v):
                if self.betw[v] >= max_betw:
                    max_betw = self.betw[v]
                    designated_cache = v
        # Forward content
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v, size=size, priority=priority)
            if v == designated_cache:
                self.controller.put_content(v, size=size, priority=priority)
        self.controller.end_session()


@register_strategy("RAND_BERNOULLI")
class RandomBernoulli(Strategy):
    """Bernoulli random cache insertion.

    In this strategy, a content is randomly inserted in a cache on the path
    from serving node to receiver with probability *p*.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, p=0.2, **kwargs):
        super().__init__(view, controller)
        self.p = p

    @inheritdoc(Strategy)
    def  process_event(self, time, receiver, content, size, priority, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log, priority)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v, size=size, priority=priority):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v, size=size, priority=priority)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v, size=size, priority=priority)
            if v != receiver and self.view.has_cache(v):
                if random.random() < self.p:
                    self.controller.put_content(v, size=size, priority=priority)
        self.controller.end_session()


@register_strategy("RAND_CHOICE")
class RandomChoice(Strategy):
    """Random choice strategy

    This strategy stores the served content exactly in one single cache on the
    path from serving node to receiver selected randomly.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super().__init__(view, controller)

    @inheritdoc(Strategy)
    def  process_event(self, time, receiver, content, size, priority, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log, priority)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v, size=size, priority=priority):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v, size=size, priority=priority)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        caches = [v for v in path[1:-1] if self.view.has_cache(v)]
        designated_cache = random.choice(caches) if len(caches) > 0 else None
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v, size=size, priority=priority)
            if v == designated_cache:
                self.controller.put_content(v, size=size, priority=priority)
        self.controller.end_session()

   
@register_strategy("COST")
class Cost(Strategy):
    """Cost strategy 

    This strategy caches content objects based on a cost function.
    The cost function includes:
        storage cost: depreciation + energy
        retrieval cost: bandwidth + energy + QoS penalty
    
    Storage gain == cost of retrieval from closest node des:
    Storage loss == storage cost + min (storage gain of content to evict)
    
    If (storage gain > storage loss) :
        cache data
    Else : 
        don't cache

    Since the nodes are multi-tier, we'll find which tiers are going to have a write on them using QM-ARC.    
         
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super().__init__(view, controller)
        self.cache_size = view.cache_nodes(size=True)
        self.penalty_table = kwargs['penalty_table']
        self.tiers = kwargs['tiers']
        for _, value in self.cache_size.items():
            for tier in self.tiers:
                tier['actual_size'] = round(tier['size_factor'] * value)
            if self.tiers[0]['actual_size'] == 0:
                self.tiers[0]['actual_size'] = 1
                for i in range(1, len(self.tiers)):
                    if self.tiers[i]['actual_size'] > 0:
                        self.tiers[i]['actual_size'] -= 1
                        break
            self.tiers = [tier for tier in self.tiers if tier['actual_size'] > 0]
        self.cost_per_joule = kwargs['cost_per_joule']
        self.cost_per_bit = kwargs['cost_per_bit']
        self.router_energy_density = kwargs['router_energy_density']
        self.link_energy_density = kwargs['link_energy_density']
        self.clf, self.feature_names, self.label_encoder_content = self.loadmodel("/home/lydia/icarus/examples/lce-vs-probcache/model")
        self.predictions = defaultdict(list)
        
    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, size, priority, log):
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        self.controller.start_session(time, receiver, content, log, priority)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v, size=size, priority=priority):
                    serving_node = v
                    break
        else:
            self.controller.get_content(v, tier_index=0, size=size, priority=priority)
            serving_node = v
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v, main_path=True, size=size, priority=priority)
            # we are at node v
            if self.view.has_cache(v):
                if content in self.predictions:
                    is_reaccessed = self.predictions[content]
                else:
                    is_reaccessed = self._predict_event(time, content, size, priority)
                if is_reaccessed:
                    logger.info("popular")
                    cache_dump_t1, cache_dump_t2, b2, p, cache_lenghth = self.view.cache_dump(v)
                    t1 = list(cache_dump_t1.keys()) if cache_dump_t1 else []
                    t2 = list(cache_dump_t2.keys()) if cache_dump_t2 else []
                    if t1 and ((content in b2 and len(t1) == p) or (len(t1) > p)):
                        cache_dump = t1
                        cache_dump_dict = cache_dump_t1
                    else:
                        cache_dump = t2
                        cache_dump_dict = cache_dump_t2

                    if cache_lenghth == self.cache_size[v]:
                        paths = {}
                        for c in cache_dump[:max(2, round(0.1 * len(cache_dump)))]:
                            c_size = cache_dump_dict[c][1]
                            c_priority = cache_dump_dict[c][2]
                            # c_is_reaccessed = self._predict_event(time, content, size, priority)
                            # if c_is_reaccessed:
                            # logger.info(f"content: {c} will be popular")
                            # look for source of data c in node v
                            c_source = self.view.content_source(c)
                            c_path = self.view.shortest_path(v, c_source)
                            for c_u, c_v in path_links(c_path):
                                if self.view.has_cache(c_v):
                                    if self.view.cache_lookup(c_v, c):
                                        c_serving_node = c_v
                                        break
                            else:
                                c_serving_node = c_v
                        
                            c_gain = self.storage_gain(list(reversed(self.view.shortest_path(v, c_serving_node))), c_size, c_priority)
                            paths[c] = c_gain
                        if paths: 
                            # we choose the data with the least retrieval time to evict
                            min_content, min_gain = min(paths.items(), key=lambda x: x[1])
                            min_size = cache_dump_dict[min_content][1]
                            min_priority = cache_dump_dict[min_content][2]
                            # calculate storage loss
                            storage_loss = self.storage_loss(v, content, size, min_gain)
                            # calculate storage gain
                            storage_gain = self.storage_gain(list(reversed(self.view.shortest_path(v, serving_node))), size, priority) 
                            logger.info(f"storage_loss:{storage_loss}")
                            logger.info(f"storage_gain:{storage_gain}")
                            self.print_costs(list(reversed(self.view.shortest_path(v, serving_node))), priority, v, receiver, content,size, min_gain, min_content, min_size, min_priority)
                            if storage_gain > storage_loss:
                                logger.info("storage_gain > storage_loss")
                                tier_index = self.controller.get_tier_index(v, content)
                                self.controller.put_content(v, min_content=min_content, tier_index=tier_index, size=size, priority=priority)
                            else:
                                logger.info("cost is not for it")
                        else:
                            logger.info("no paths")
                            tier_index = self.controller.get_tier_index(v, content)
                            self.controller.put_content(v, tier_index=tier_index, size=size, priority=priority)
                    else:
                        tier_index = self.controller.get_tier_index(v, content)
                        self.controller.put_content(v, tier_index=tier_index, size=size, priority=priority)
                else:
                    logger.info("other")
        self.controller.end_session()
    
    def loadmodel(self, model_filename):
        clffile = model_filename + "/clf.pkl"
        encoderfile = model_filename + "/labelencoder.pkl"
        modelparams = model_filename + "/modelparams.csv"
        with open(clffile, 'rb') as f:
            clf = pickle.load(f)
        with open(encoderfile, 'rb') as f:
            label_encoder_content = pickle.load(f)
        with open(modelparams, 'r') as params:
            reader = csv.reader(params)
            next(reader)  # Skip the header
            for row in reader:
                feature_names = row
                
        return clf, feature_names, label_encoder_content 

    def storage_gain(self, path, content_size, priority) -> float:
        # storage_gain = self.bandwidth_cost(path, content_size) + self.transmission_energy_cost(path, content_size) + self.penalty_cost(path, priority)    
        
        storage_gain = self.transmission_energy_cost(path, content_size) + self.penalty_cost(path, priority)    
        return storage_gain
    
    def storage_loss(self, receiver, content, content_size, min_value) -> float:
        tier_index = self.controller.get_tier_index(receiver, content)
        # cost_storage = self.depreciation_cost(tier_index, content_size) + self.storage_energy_cost(tier_index, receiver, content_size)
        cost_storage = self.storage_energy_cost(tier_index, receiver, content_size)
        storage_loss =  cost_storage + min_value    
        return storage_loss
        
    def depreciation_cost(self, tier_index, content_size) -> float: 
        depreciation_cost = 0.0
        tier = self.tiers[0]
        tier_max_capacity = tier['actual_size']
        tier_purchase_cost = tier['purchase_cost']
        tier_lifespan = tier['lifespan'] * 365 * 24 * 60 * 60
        depreciation_cost += (content_size * tier_purchase_cost) / (tier_lifespan * tier_max_capacity)
        for tier in self.tiers[tier_index:]:
            tier_max_capacity = tier['actual_size']
            tier_purchase_cost = tier['purchase_cost']
            tier_lifespan = tier['lifespan'] * 365 * 24 * 60 * 60
            depreciation_cost += (content_size * tier_purchase_cost) / (tier_lifespan * tier_max_capacity)
        return depreciation_cost

    def storage_energy_cost(self, tier_index, receiver, content_size) -> float:
        tiers_last_access = self.view.get_last_access(receiver)
        
        tier = self.tiers[tier_index]
        tier_active_power_density  = tier['active_caching_power_density']
        tier_idle_power_density = tier['idle_power_density']
       
        idle_time = max(0.0, time.time() - tiers_last_access[tier_index])

        read_time = tier['latency'] + content_size / tier['read_throughput']
        energy_cost = ((tier_idle_power_density * idle_time) + (tier_active_power_density * read_time * content_size)) * self.cost_per_joule
        
        for i, tier in enumerate(self.tiers[tier_index:], start=tier_index):
            tier_active_power_density  = tier['active_caching_power_density']
            tier_idle_power_density = tier['idle_power_density']
            
            idle_time = max(0.0, time.time() - tiers_last_access[i])
            
            write_time = tier['latency'] + content_size / tier['write_throughput']
            energy_cost += ((tier_idle_power_density * idle_time) + (tier_active_power_density * write_time * content_size)) * self.cost_per_joule
        
        return energy_cost 

    def penalty_cost(self, path, priority) -> float:
        latency = 2 * sum(self.view.link_delay(u, v) for u, v in path_links(path))
        for entry in self.penalty_table:
            if latency <= entry["delay"]:
                if priority == "high":
                    return entry["P0"] * 1e-8 
                elif priority == "low":
                    return entry["P1"] * 1e-8 
                return

        # If no penalty threshold matches, raise an error (this should not happen)
        raise ValueError(f"Latency {latency} is out of range for priority {priority}.")
        
    def transmission_energy_cost(self, path, content_size) -> float:
        nodes_energy_cost = (len(path) + 1) * content_size * self.router_energy_density * self.cost_per_joule
        links_energy_cost = len(path) * content_size * self.link_energy_density * self.cost_per_joule        
        return nodes_energy_cost + links_energy_cost

    def bandwidth_cost(self, path, content_size) -> float:
        return len(path) * content_size * self.cost_per_bit

    def print_costs(self, path, priority, node, receiver, content, content_size, min_value, min_content, min_size, min_priority):
        tier_index = self.controller.get_tier_index(node, content)
        depreciation_cost = 0.0
        tiers_last_access = self.view.get_last_access(node)
        tier = self.tiers[tier_index]
        tier_active_power_density  = tier['active_caching_power_density']
        tier_idle_power_density = tier['idle_power_density']
        idle_time = max(0.0, time.time() - tiers_last_access[tier_index])
        read_time = tier['latency'] + content_size / tier['read_throughput']
        energy_cost = ((tier_idle_power_density * idle_time) + (tier_active_power_density * read_time * content_size)) * self.cost_per_joule
        tier_max_capacity = tier['actual_size']
        tier_purchase_cost = tier['purchase_cost']
        tier_lifespan = tier['lifespan'] * 365 * 24 * 60 * 60
        depreciation_cost += (content_size * tier_purchase_cost) / (tier_lifespan * tier_max_capacity)
        
        for i, tier in enumerate(self.tiers[tier_index:], start=tier_index):
            tier_max_capacity = tier['actual_size']
            tier_purchase_cost = tier['purchase_cost']
            tier_lifespan = tier['lifespan'] * 365 * 24 * 60 * 60
            tier_active_power_density  = tier['active_caching_power_density']
            tier_idle_power_density = tier['idle_power_density']
            idle_time = max(0.0, time.time() - tiers_last_access[i])
            write_time = tier['latency'] + content_size / tier['write_throughput']
            energy_cost += ((tier_idle_power_density * idle_time) + (tier_active_power_density * write_time * content_size)) * self.cost_per_joule
            depreciation_cost += (content_size * tier_purchase_cost) / (tier_lifespan * tier_max_capacity)
        
        bandwidth_cost = len(path) * content_size * self.cost_per_bit
        nodes_energy_cost = (len(path) + 1) * content_size * self.router_energy_density * self.cost_per_joule
        links_energy_cost = len(path) * content_size * self.link_energy_density * self.cost_per_joule
        latency = 2 * sum(self.view.link_delay(u, v) for u, v in path_links(path))
        for entry in self.penalty_table:
            if latency <= entry["delay"]:
                if priority == "high":
                    penalty = entry["P0"] * 1e-8 
                elif priority == "low":
                    penalty = entry["P1"] * 1e-8 
        total = depreciation_cost + energy_cost + bandwidth_cost + nodes_energy_cost + links_energy_cost + penalty
        
        logger.info(f"content:{content},content_size:{content_size},priority:{priority}, min_content:{min_content},min_size:{min_size},min_priority:{min_priority}")
        logger.info(f"storage_loss = depreciation_cost :{depreciation_cost} + storage_energy_cost :{energy_cost} + min_value:{min_value} = {depreciation_cost+ energy_cost + min_value}")
        logger.info(f"storage_gain = bandwidth_cost :{bandwidth_cost} + transmission_energy_cost :{nodes_energy_cost + links_energy_cost} + penalty : {penalty} = {bandwidth_cost + nodes_energy_cost + links_energy_cost + penalty} ")    
        logger.info("ESTIMATED: node:%s,receiver:%s, transmission_energy_cost %s, penalty:%s, total:%s "%
                    (node, receiver, nodes_energy_cost + links_energy_cost, penalty, total))           
   
    def _predict_event(self, time, content, size, priority):
        # Create a DataFrame for the new event
        event_df = pd.DataFrame([(time, content, size, priority)], columns=['timestamp', 'content', 'size', 'priority'])
        
        event_df['priority'] = event_df['priority'].map({'low': 0, 'high': 1})
        # If the new 'content' is unseen, fit the LabelEncoder on both the original and new data
        if content not in self.label_encoder_content.classes_:
            # Extend the classes in the label encoder to include new content
            new_classes = np.append(self.label_encoder_content.classes_, content)
            self.label_encoder_content.classes_ = new_classes

        event_df['content'] = event_df['content'].astype(str)
        event_df['content'] = self.label_encoder_content.fit_transform([content])[0]
        event_df['size'] = pd.to_numeric(event_df['size'], errors='coerce')
        event_df['inter_arrival_time'] = 0
        event_df['prev_access_count'] = 0
        event_df['time_since_last_access'] = 0

        event_df = event_df.reindex(columns=self.feature_names, fill_value=0)

        # Predict reaccess
        is_reaccessed = self.clf.predict(event_df)[0]
        self.predictions[content].append(is_reaccessed)
        return is_reaccessed
    
    def _process_chunk(self):
        df = pd.DataFrame(self.events, columns=['timestamp', 'receiver', 'content', 'priority'])
        self.events = []
        df['is_reaccessed'] = df.duplicated(subset='content', keep=False).astype(int)
        df['priority'] = df['priority'].map({'low': 0, 'high': 1})
        df = pd.get_dummies(df, columns=['receiver'], drop_first=True)

        X = df.drop(['is_reaccessed', 'timestamp'], axis=1)
        y = df['is_reaccessed']

        # Ensure there is data to process
        if X.empty or y.empty:
            return

        # Train and evaluate model in parallel
        try:
            results = Parallel(n_jobs=-1)(
                delayed(self._train_and_evaluate_model)(X, y)
            )

            self.is_model_trained = True

        except ValueError as e:
            print(f"Error in _process_chunk: {e}")
    
    def _train_and_evaluate_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
        self.xgboost_model.fit(X_train, y_train)
        y_pred = self.xgboost_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        return accuracy, precision, recall, f1
    
@register_strategy("LRUCOST")
class LRUCost(Strategy):
    """Cost strategy 

    This strategy caches content objects based on a cost function.
    The cost function includes:
        storage cost: depreciation + energy
        retrieval cost: bandwidth + energy + QoS penalty
    
    Storage gain == cost of retrieval from closest node des:
    Storage loss == storage cost + min (storage gain of content to evict)
    
    If (storage gain > storage loss) :
        cache data
    Else : 
        don't cache

    Since the nodes are multi-tier, we'll find which tiers are going to have a write on them using QM-ARC.    
         
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super().__init__(view, controller)
        self.cache_size = view.cache_nodes(size=True)
        self.penalty_table = kwargs['penalty_table']
        self.tiers = kwargs['tiers']
        for _, value in self.cache_size.items():
            for tier in self.tiers:
                tier['actual_size'] = round(tier['size_factor'] * value)
            if self.tiers[0]['actual_size'] == 0:
                self.tiers[0]['actual_size'] = 1
                for i in range(1, len(self.tiers)):
                    if self.tiers[i]['actual_size'] > 0:
                        self.tiers[i]['actual_size'] -= 1
                        break
            self.tiers = [tier for tier in self.tiers if tier['actual_size'] > 0]
        self.cost_per_joule = kwargs['cost_per_joule']
        self.cost_per_bit = kwargs['cost_per_bit']
        self.router_energy_density = kwargs['router_energy_density']
        self.link_energy_density = kwargs['link_energy_density']
        self.clf, self.feature_names, self.label_encoder_content = self.loadmodel("/home/lydia/icarus/examples/lce-vs-probcache/model")
        self.predictions = defaultdict(list)
        
    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, size, priority, log):
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        self.controller.start_session(time, receiver, content, log, priority)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v, size=size, priority=priority):
                    serving_node = v
                    break
        else:
            self.controller.get_content(v, tier_index=0, size=size, priority=priority)
            serving_node = v
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v, main_path=True, size=size, priority=priority)
            # we are at node v
            if self.view.has_cache(v):
                if content in self.predictions:
                    is_reaccessed = self.predictions[content]
                else:
                    is_reaccessed = self._predict_event(time, content, size, priority)
                if is_reaccessed:
                    cache_dump = self.view.cache_dump(v)
                    if len(cache_dump) == self.cache_size[v]:
                        paths = {}
                        for c in cache_dump:
                            c_size = cache_dump[c][0]
                            c_priority = cache_dump[c][1]
                            c_source = self.view.content_source(c)
                            c_path = self.view.shortest_path(v, c_source)
                            for c_u, c_v in path_links(c_path):
                                if self.view.has_cache(c_v) and self.view.cache_lookup(c_v, c):
                                    c_serving_node = c_v
                                    break
                            else:
                                c_serving_node = c_v
                            
                            c_path_reversed = list(reversed(self.view.shortest_path(v, c_serving_node)))
                            c_gain = self.storage_gain(c_path_reversed, c_size, c_priority)
                            paths[c] = c_gain
                        if paths:
                            min_content, min_gain = min(paths.items(), key=lambda x: x[1])
                            min_size = cache_dump[min_content][0]
                            min_priority = cache_dump[min_content][1]
                            storage_loss = self.storage_loss(v, content, size, min_gain)
                            storage_gain = self.storage_gain(list(reversed(self.view.shortest_path(v, serving_node))), size, priority) 
                            if storage_gain > storage_loss:
                                logger.info("cache thanks to costs")
                                tier_index = self.controller.get_tier_index(v, content)
                                self.controller.put_content(v, min_content=(min_content, min_size, min_priority), tier_index=tier_index, size=size, priority=priority)
                            else:
                                logger.info("cost is not for it")
                        else:
                            tier_index = self.controller.get_tier_index(v, content)
                            self.controller.put_content(v, tier_index=tier_index, size=size, priority=priority)
                    else:
                        tier_index = self.controller.get_tier_index(v, content)
                        self.controller.put_content(v, tier_index=tier_index, size=size, priority=priority)
        self.controller.end_session()

    def loadmodel(self, model_filename):
        clffile = model_filename + "/clf.pkl"
        encoderfile = model_filename + "/labelencoder.pkl"
        modelparams = model_filename + "/modelparams.csv"
        with open(clffile, 'rb') as f:
            clf = pickle.load(f)
        with open(encoderfile, 'rb') as f:
            label_encoder_content = pickle.load(f)
        with open(modelparams, 'r') as params:
            reader = csv.reader(params)
            next(reader)  # Skip the header
            for row in reader:
                feature_names = row
                
        return clf, feature_names, label_encoder_content 

    def storage_gain(self, path, content_size, priority) -> float:
        storage_gain = self.bandwidth_cost(path, content_size) + self.transmission_energy_cost(path, content_size) + self.penalty_cost(path, priority)    
        return storage_gain
    
    def storage_loss(self, receiver, content, content_size, min_value) -> float:
        tier_index = self.controller.get_tier_index(receiver, content)
        storage_loss = self.depreciation_cost(tier_index, content_size) + self.storage_energy_cost(tier_index, receiver, content_size) + min_value    
        return storage_loss
        
    def depreciation_cost(self, tier_index, content_size) -> float: 
        depreciation_cost = 0.0
        tier = self.tiers[0]
        tier_max_capacity = tier['actual_size']
        tier_purchase_cost = tier['purchase_cost']
        tier_lifespan = tier['lifespan'] * 365 * 24 * 60 * 60
        depreciation_cost += (content_size * tier_purchase_cost) / (tier_lifespan * tier_max_capacity)
        for tier in self.tiers[tier_index:]:
            tier_max_capacity = tier['actual_size']
            tier_purchase_cost = tier['purchase_cost']
            tier_lifespan = tier['lifespan'] * 365 * 24 * 60 * 60
            depreciation_cost += (content_size * tier_purchase_cost) / (tier_lifespan * tier_max_capacity)
        return depreciation_cost

    def storage_energy_cost(self, tier_index, receiver, content_size) -> float:
        tiers_last_access = self.view.get_last_access(receiver)
        
        tier = self.tiers[tier_index]
        tier_active_power_density  = tier['active_caching_power_density']
        tier_idle_power_density = tier['idle_power_density']
       
        idle_time = max(0.0, time.time() - tiers_last_access[tier_index])

        read_time = tier['latency'] + content_size / tier['read_throughput']
        energy_cost = ((tier_idle_power_density * idle_time) + (tier_active_power_density * read_time * content_size)) * self.cost_per_joule
        
        for i, tier in enumerate(self.tiers[tier_index:], start=tier_index):
            tier_active_power_density  = tier['active_caching_power_density']
            tier_idle_power_density = tier['idle_power_density']
            
            idle_time = max(0.0, time.time() - tiers_last_access[i])
            
            write_time = tier['latency'] + content_size / tier['write_throughput']
            energy_cost += ((tier_idle_power_density * idle_time) + (tier_active_power_density * write_time * content_size)) * self.cost_per_joule
        
        return energy_cost 

    def penalty_cost(self, path, priority) -> float:
        latency = 2 * sum(self.view.link_delay(u, v) for u, v in path_links(path))
        for entry in self.penalty_table:
            if latency <= entry["delay"]:
                if priority == "high":
                    return entry["P0"] * 1e-8 
                elif priority == "low":
                    return entry["P1"] * 1e-8 
                return

        # If no penalty threshold matches, raise an error (this should not happen)
        raise ValueError(f"Latency {latency} is out of range for priority {priority}.")
        
    def transmission_energy_cost(self, path, content_size) -> float:
        nodes_energy_cost = (len(path) + 1) * content_size * self.router_energy_density * self.cost_per_joule
        links_energy_cost = len(path) * content_size * self.link_energy_density * self.cost_per_joule        
        return nodes_energy_cost + links_energy_cost

    def bandwidth_cost(self, path, content_size) -> float:
        return len(path) * content_size * self.cost_per_bit

    def print_costs(self, path, priority, node, receiver, content, content_size, min_value, min_content, min_size, min_priority):
        tier_index = self.controller.get_tier_index(node, content)
        depreciation_cost = 0.0
        tiers_last_access = self.view.get_last_access(node)
        tier = self.tiers[tier_index]
        tier_active_power_density  = tier['active_caching_power_density']
        tier_idle_power_density = tier['idle_power_density']
        idle_time = max(0.0, time.time() - tiers_last_access[tier_index])
        read_time = tier['latency'] + content_size / tier['read_throughput']
        energy_cost = ((tier_idle_power_density * idle_time) + (tier_active_power_density * read_time * content_size)) * self.cost_per_joule
        tier_max_capacity = tier['actual_size']
        tier_purchase_cost = tier['purchase_cost']
        tier_lifespan = tier['lifespan'] * 365 * 24 * 60 * 60
        depreciation_cost += (content_size * tier_purchase_cost) / (tier_lifespan * tier_max_capacity)
        
        for i, tier in enumerate(self.tiers[tier_index:], start=tier_index):
            tier_max_capacity = tier['actual_size']
            tier_purchase_cost = tier['purchase_cost']
            tier_lifespan = tier['lifespan'] * 365 * 24 * 60 * 60
            tier_active_power_density  = tier['active_caching_power_density']
            tier_idle_power_density = tier['idle_power_density']
            idle_time = max(0.0, time.time() - tiers_last_access[i])
            write_time = tier['latency'] + content_size / tier['write_throughput']
            energy_cost += ((tier_idle_power_density * idle_time) + (tier_active_power_density * write_time * content_size)) * self.cost_per_joule
            depreciation_cost += (content_size * tier_purchase_cost) / (tier_lifespan * tier_max_capacity)
        
        bandwidth_cost = len(path) * content_size * self.cost_per_bit
        nodes_energy_cost = (len(path) + 1) * content_size * self.router_energy_density * self.cost_per_joule
        links_energy_cost = len(path) * content_size * self.link_energy_density * self.cost_per_joule
        latency = 2 * sum(self.view.link_delay(u, v) for u, v in path_links(path))
        for entry in self.penalty_table:
            if latency <= entry["delay"]:
                if priority == "high":
                    penalty = entry["P0"] * 1e-8 
                elif priority == "low":
                    penalty = entry["P1"] * 1e-8 
        total = depreciation_cost + energy_cost + bandwidth_cost + nodes_energy_cost + links_energy_cost + penalty
        
        logger.info(f"content:{content},content_size:{content_size},priority:{priority}, min_content:{min_content},min_size:{min_size},min_priority:{min_priority}")
        logger.info(f"storage_loss = depreciation_cost :{depreciation_cost} + storage_energy_cost :{energy_cost} + min_value:{min_value} = {depreciation_cost+ energy_cost + min_value}")
        logger.info(f"storage_gain = bandwidth_cost :{bandwidth_cost} + transmission_energy_cost :{nodes_energy_cost + links_energy_cost} + penalty : {penalty} = {bandwidth_cost + nodes_energy_cost + links_energy_cost + penalty} ")    
        logger.info("ESTIMATED: node:%s,receiver:%s, transmission_energy_cost %s, penalty:%s, total:%s "%
                    (node, receiver, nodes_energy_cost + links_energy_cost, penalty, total))           
   
    def _predict_event(self, time, content, size, priority):
        # Create a DataFrame for the new event
        event_df = pd.DataFrame([(time, content, size, priority)], columns=['timestamp', 'content', 'size', 'priority'])
        
        event_df['priority'] = event_df['priority'].map({'low': 0, 'high': 1})
        # If the new 'content' is unseen, fit the LabelEncoder on both the original and new data
        if content not in self.label_encoder_content.classes_:
            # Extend the classes in the label encoder to include new content
            new_classes = np.append(self.label_encoder_content.classes_, content)
            self.label_encoder_content.classes_ = new_classes

        event_df['content'] = event_df['content'].astype(str)
        event_df['content'] = self.label_encoder_content.fit_transform([content])[0]
        event_df['size'] = pd.to_numeric(event_df['size'], errors='coerce')
        event_df['inter_arrival_time'] = 0
        event_df['prev_access_count'] = 0
        event_df['time_since_last_access'] = 0

        event_df = event_df.reindex(columns=self.feature_names, fill_value=0)

        # Predict reaccess
        is_reaccessed = self.clf.predict(event_df)[0]
        self.predictions[content].append(is_reaccessed)
        return is_reaccessed
    
    def _process_chunk(self):
        df = pd.DataFrame(self.events, columns=['timestamp', 'receiver', 'content', 'priority'])
        self.events = []
        df['is_reaccessed'] = df.duplicated(subset='content', keep=False).astype(int)
        df['priority'] = df['priority'].map({'low': 0, 'high': 1})
        df = pd.get_dummies(df, columns=['receiver'], drop_first=True)

        X = df.drop(['is_reaccessed', 'timestamp'], axis=1)
        y = df['is_reaccessed']

        # Ensure there is data to process
        if X.empty or y.empty:
            return

        # Train and evaluate model in parallel
        try:
            results = Parallel(n_jobs=-1)(
                delayed(self._train_and_evaluate_model)(X, y)
            )

            self.is_model_trained = True

        except ValueError as e:
            print(f"Error in _process_chunk: {e}")
    
    def _train_and_evaluate_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
        self.xgboost_model.fit(X_train, y_train)
        y_pred = self.xgboost_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        return accuracy, precision, recall, f1
    