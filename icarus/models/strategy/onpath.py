"""Implementations of all on-path strategies"""
from collections import defaultdict
import csv
import json
import logging
import os
from pathlib import Path
import pickle
import math
import random
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    "CacheLessForMore",
    "RandomBernoulli",
    "RandomChoice",
    "CacheLessToSaveMore",
    "CPCacheCooperativeCaching",
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
            self.controller.forward_content_hop(u, v, main_path=True, size=size, priority=priority)
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
            self.controller.forward_content_hop(u, v, main_path=True, size=size, priority=priority)
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
            self.controller.forward_content_hop(u, v, main_path=True, size=size, priority=priority)
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
            self.controller.forward_content_hop(u, v, main_path=True, size=size, priority=priority)
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
            self.controller.forward_content_hop(u, v, main_path=True, size=size, priority=priority)
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
            self.controller.forward_content_hop(u, v, main_path=True, size=size, priority=priority)
            if v == designated_cache:
                self.controller.put_content(v, size=size, priority=priority)
        self.controller.end_session()

   
@register_strategy("CL2SM")
class CacheLessToSaveMore(Strategy):
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
        self.req_size = 150
        self.cache_size = view.cache_nodes(size=True)

        self.penalty_table = sorted(kwargs['penalty_table'], key=lambda e: e["delay"])
        self.cost_per_joule = kwargs['cost_per_joule']
        self.cost_per_bit = kwargs['cost_per_bit']
        self.router_energy_density = kwargs['router_energy_density']
        self.link_energy_density = kwargs['link_energy_density']

        self.gain_per_data = {}
        self.request_counter = {}
        self.log_file_path = '../../examples/lce-vs-probcache/path_log.csv'

    def restore_strategy_state(self, strategy_name=None, period=None):
        """
        Restore the saved state (gain_per_data, request_counter) for this strategy.
        Supports period-specific resumes (e.g. CL2SM_p2.pkl).
        """
        strategy_name = strategy_name or self.__class__.__name__
        saved_dir = Path("strategy_states")

        # Choose filename based on period number if provided
        filename = f"{strategy_name}_p{period}.pkl"
        saved_state_path = saved_dir / filename
        saved_json_path = saved_state_path.with_suffix(".json")

        restored = False
        if saved_state_path.exists():
            try:
                with open(saved_state_path, "rb") as f:
                    state = pickle.load(f)
                self.gain_per_data = state.get("gain_per_data", {})
                self.request_counter = state.get("request_counter", {})
                print(f"[♻️] Restored {strategy_name} (period={period or 'latest'}) from {saved_state_path}")
                restored = True
            except Exception as e:
                print(f"[⚠️] Failed to restore {strategy_name} (pkl): {e}")

        elif saved_json_path.exists():
            try:
                with open(saved_json_path, "r") as jf:
                    state = json.load(jf)
                self.gain_per_data = state.get("gain_per_data", {})
                self.request_counter = state.get("request_counter", {})
                print(f"[♻️] Restored {strategy_name} (period={period or 'latest'}) from {saved_json_path}")
                restored = True
            except Exception as e:
                print(f"[⚠️] Failed to restore {strategy_name} (json): {e}")

        if not restored:
            print(f"[ℹ️] No previous strategy state found for {strategy_name} (cold start).")

    def save_strategy_state(self, strategy_name=None, period=None):
        """
        Save the current strategy state to disk, using period-specific filenames.
        """
        strategy_name = strategy_name or self.__class__.__name__
        saved_dir = Path("strategy_states")
        saved_dir.mkdir(exist_ok=True)

        filename = f"{strategy_name}_p{period}.pkl" if period else f"{strategy_name}_state.pkl"
        filepath = saved_dir / filename

        state = {
            "gain_per_data": getattr(self, "gain_per_data", {}),
            "request_counter": getattr(self, "request_counter", {}),
        }

        # Save as pickle
        try:
            with open(filepath, "wb") as f:
                pickle.dump(state, f)
            print(f"[💾] Saved strategy state to {filepath}")
        except Exception as e:
            print(f"[⚠️] Failed to save {strategy_name} state (pkl): {e}")

        # Also save JSON copy
        jsonpath = filepath.with_suffix(".json")
        try:
            with open(jsonpath, "w") as jf:
                json.dump(state, jf, indent=2)
            print(f"[📄] JSON copy saved to {jsonpath}")
        except Exception as e:
            print(f"[⚠️] Failed to save {strategy_name} state (json): {e}")

    def _tiers(self, node):
        try:
            # New API: view.cache_tiers(node) -> list[tiers] for that node
            tiers = self.view.cache_tiers(node)
        except TypeError:
            # Legacy API fallback: view.cache_tiers() -> dict[node]->list[tiers]
            all_tiers = self.view.cache_tiers()
            tiers = all_tiers.get(node, [])
        return tiers or []
    
    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, size, priority, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
         # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log, priority)
        serving_node = None
        for u, v in path_links(path):
            if v not in self.gain_per_data:
                self.gain_per_data[v] = {}
            if v not in self.request_counter:
                self.request_counter[v] = {}
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v, size=size, priority=priority):
                    serving_node = v
                    break
        if serving_node is None:
            # No cache hits, get content from source
            self.controller.get_content(source, tier_index=0, size=size, priority=priority, time=time)
            serving_node = source
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v, main_path=True, size=size, priority=priority)
            
            if not self.view.has_cache(v):
                continue
            
            tiers = self._tiers(v)
            if not tiers:
                continue
            
            new_value = self.request_counter[v].get(content) + 1 if self.request_counter[v].get(content) else 1
            self.request_counter[v].update({content: new_value})
            reaccess_prob = self.get_probability_estimate(content, self.request_counter[v]) 
            # print(f"is_reaccessed probability for content {content} at node {v}: {is_reaccessed}")
            # print(f"reaccess_prob for content {content} at node {v}: {reaccess_prob}")
            # print(f"is_reaccessed:{is_reaccessed}")
            # if is_reaccessed:
            new_path = list(reversed(self.view.shortest_path(v, serving_node)))
            storage_gain, band, trans, pen = self.storage_gain(new_path, size, priority)
            adjusted_gain = storage_gain * reaccess_prob
            self.gain_per_data[v].update({content: adjusted_gain})
            
            cache_dump = self.view.cache_dump(v, k=content)
            # print(f"cache_dump:{cache_dump}")
            if len(cache_dump) == self.cache_size[v]:
                paths = {}
                for c in list(cache_dump.keys())[:max(2, round(0.1 * len(cache_dump)))]:
                    # paths[c] = self.gain_per_data[v].get(c) * np.exp(-self.rate * self.sess_count)
                    # self.gain_per_data[v].update({c: paths[c]})
                    paths[c] = self.gain_per_data[v].get(c)
                if paths:
                    # we choose the data with the least retrieval time to evict
                    min_content, min_gain = min(paths.items(), key=lambda x: x[1])
                    # calculate storage loss
                    storage_loss, dep, stor = self.storage_loss(v, tiers, content, size, priority) 
                    baseline_min_gain = self.gain_per_data[v].get(min_content, 0.0) if min_content else 0.0
                    
                    if adjusted_gain >= storage_loss + baseline_min_gain:
                        logger.info("storage_gain > storage_loss")
                        tier_index = self.controller.get_tier_index(v, content, priority)
                        self.controller.put_content(v, min_content=min_content, tier_index=tier_index, size=size, priority=priority)
                    else:
                        logger.info("cost is not for it")
                        for node, value in self.gain_per_data.items():
                            logger.info(f"node:{node}, value:{value}")
                else:
                    logger.info("no paths")
                    tier_index = self.controller.get_tier_index(v, content, priority)
                    self.controller.put_content(v, tier_index=tier_index, size=size, priority=priority)
            else:
                tier_index = self.controller.get_tier_index(v, content, priority)
                self.controller.put_content(v, tier_index=tier_index, size=size, priority=priority) 
        
        self.controller.end_session()

    def get_probability_estimate(self, content, request_count):
        total = sum(request_count.values())
        return request_count[content] / total if total > 0 else 0.0
        
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
        band = self.bandwidth_cost(path, content_size) 
        trans = self.transmission_energy_cost(path, content_size)
        pen = self.penalty_cost(path, priority)  
        storage_gain = band + trans + pen
        # logger.info(f"ESTIMATED cost_bandwidth:{band},cost_penalty:{pen}, cost_transmission:{trans} = storage_gain {storage_gain}") 
        return storage_gain, band, trans, pen
    
    def storage_loss(self, receiver, tiers, content, content_size, content_priority) -> float:
        tier_index = self.controller.get_tier_index(receiver, content, content_priority)
        dep = self.depreciation_cost(tiers, tier_index, content_size)
        stor = self.storage_energy_cost(tiers, tier_index, receiver, content_size)
        storage_loss = dep + stor
        # logger.info(f"ESTIMATED cost_depreciation:{dep}, cost_storage:{stor} = storage_loss {storage_loss}") 
        return storage_loss, dep, stor
        
    def depreciation_cost(self, tiers, tier_index, content_size) -> float: 
        depreciation_cost = 0.0
        for tier in tiers[tier_index:]:
            tier_max_capacity = tier['actual_size_bytes']
            tier_purchase_cost = tier['purchase_cost']
            tier_lifespan = tier['lifespan'] * 365 * 24 * 60 * 60
            depreciation_cost += (content_size * tier_purchase_cost) / (tier_lifespan * tier_max_capacity)
        
        if tier_index != 0 and len(tiers) > 1:
            tier = tiers[tier_index]
            tier_max_capacity = tier['actual_size_bytes']
            tier_purchase_cost = tier['purchase_cost']
            tier_lifespan = tier['lifespan'] * 365 * 24 * 60 * 60
            read_cost = (content_size * tier_purchase_cost) / (tier_lifespan * tier_max_capacity)
            depreciation_cost += read_cost
        return depreciation_cost

    def storage_energy_cost(self, tiers, tier_index, receiver, content_size) -> float:
        read_cost = 0.0
        write_cost = 0.0
        tier = tiers[tier_index]
        tier_max_capacity = tier['actual_size_bytes']
        tier_active_power_density  = tier['active_caching_power_density']
        tier_idle_power_density = tier['idle_power_density_per_bit']
        
        tiers_last_access = self.view.get_tiers_last_access(receiver)
        idle_time = 0.0 if tiers_last_access[tier_index]==0 else time.time() - tiers_last_access[tier_index]
        read_time = tier['latency'] + content_size / tier['read_throughput']
        read_cost = ((tier_idle_power_density * idle_time * tier_max_capacity * 8) + (tier_active_power_density * read_time * content_size * 8)) * self.cost_per_joule
        
        for i, tier in enumerate(tiers[tier_index:], start=tier_index):
            tier_max_capacity = tier['actual_size_bytes']
            tier_active_power_density  = tier['active_caching_power_density']
            tier_idle_power_density = tier['idle_power_density_per_bit']
            idle_time = 0.0 if tiers_last_access[i]==0 else time.time() - tiers_last_access[i]
            write_time = tier['latency'] + content_size / tier['write_throughput']
            write_cost += ((tier_idle_power_density * idle_time * tier_max_capacity * 8) + (tier_active_power_density * write_time * content_size * 8)) * self.cost_per_joule
        return read_cost + write_cost

    def bandwidth_cost(self, path, content_size) -> float:
        band = 0.0
        request_path = list(reversed(path))
        data_band = (len(path) -1) * content_size * self.cost_per_bit
        req_band = (len(request_path) -1) * self.req_size * self.cost_per_bit 
        band = data_band + req_band
        # logger.info(f"path: {path},(len(path) -1):{(len(path) -1)} ,(len(request_path) -1):{(len(request_path) -1)},band:{band}, content_size:{content_size}, req_size:{self.req_size}, cost_per_bite:{self.cost_per_bit}, req_band:{req_band}, data_band:{data_band}")
        return band
    
    def transmission_energy_cost(self, path, content_size) -> float:
        request_path = list(reversed(path))
        req_nodes_energy_cost = (len(request_path) - 1) * self.req_size * self.router_energy_density * self.cost_per_joule
        req_links_energy_cost = (len(request_path) - 1) * self.req_size * self.link_energy_density * self.cost_per_joule        
        
        nodes_energy_cost = (len(path) - 1) * content_size * self.router_energy_density * self.cost_per_joule
        links_energy_cost = (len(path) - 1) * content_size * self.link_energy_density * self.cost_per_joule        
        return req_nodes_energy_cost + nodes_energy_cost + req_links_energy_cost + links_energy_cost
    
    def penalty_cost(self, path, priority) -> float:
        request_path = list(reversed(path))
        data_latency = sum(self.view.link_delay(u, v) for u, v in path_links(path))
        req_latency = sum(self.view.link_delay(u, v) for u, v in path_links(request_path))
        latency = data_latency + req_latency
        penalty_cost = 0.0
        for entry in self.penalty_table:
            if latency <= entry["delay"]:
                if priority == "high":
                    penalty_cost = entry["P0"] * 1e-8
                    break
                elif priority == "low":
                    penalty_cost = entry["P1"] * 1e-8
                    break
        return penalty_cost
    
    def _predict_event(self, time, content, size, priority):
        import numpy as np
        import pandas as pd

        # Map priority to numeric
        priority_map = {'low': 0, 'high': 1}
        priority_num = priority_map.get(priority, 0)  # default to 0 if unknown

        # Handle unseen content in label encoder:
        # XGBoost label encoder does NOT support adding new classes on the fly.
        # So if content is unseen, assign a special "unknown" label or handle gracefully.
        if content in self.label_encoder_content.classes_:
            content_encoded = self.label_encoder_content.transform([content])[0]
        else:
            # Assign an "unknown" label index or fallback to a default value
            # For example, -1 or max label + 1 (ensure consistent with training)
            # Here we use -1 as a placeholder (you may want to retrain model with unknown class)
            content_encoded = -1

        # Create DataFrame with all features expected by model
        # If your model expects additional features like inter_arrival_time, prev_access_count, time_since_last_access,
        # you need to provide them here (fill with 0 or appropriate defaults)
        data = {
            'timestamp': [time],
            'content': [content_encoded],
            'size': [size],
            'priority': [priority_num],
            'inter_arrival_time': [0],
            'prev_access_count': [0],
            'time_since_last_access': [0]
        }

        event_df = pd.DataFrame(data)

        # Reindex columns to match model's feature order, fill missing with 0
        event_df = event_df.reindex(columns=self.feature_names, fill_value=0)

        # Predict probability of re-access (class 1)
        proba = self.clf.predict_proba(event_df)[0, 1]

        # Store prediction history if needed
        self.predictions[content].append(float(proba))
        return proba


@register_strategy("CPCache")
class CPCacheCooperativeCaching(Strategy):
    
    @inheritdoc(Strategy)
    def __init__(self, view, controller,  **kwargs):
        super().__init__(view, controller)
        self.cache_size = view.cache_nodes(size=True)
        self.distance = dict(
            nx.all_pairs_dijkstra_path_length(self.view.topology(), weight="delay")
        )
        self.PName = []
        self.Global_Popularity_Count = []

        self.Local_Interest_Counter={}     
        for node in self.view.topology().nodes():
            if self.view.has_cache(node):
               self.Local_Interest_Counter[node] = {}       
               
        self.Global_Interest_Counter={}     
        for node in self.view.topology().nodes():
            if self.view.has_cache(node):
               self.Global_Interest_Counter[node] = {}            

        self.topo_degree= self.view.topology().degree
        self.degree_centrality = dict(self.topo_degree)
        self.rdegree_centrality = {key: value for key, value in self.degree_centrality.items() if self.view.has_cache(key)}       
        sources = list(self.view.topology().sources())
        receivers = list(self.view.topology().receivers())
        edge_nodes = [list(self.view.topology().neighbors(node))[0] for node in receivers]
        self.non_edge_degree_centrality = {key: value for key, value in self.rdegree_centrality.items() if key not in edge_nodes}
        self.sorted_degree_centrality = dict(sorted(self.non_edge_degree_centrality.items(), key=lambda item: item[1], reverse=True))
        dim = nx.diameter(self.view.topology())
        num_nodes = self.view.topology().number_of_nodes()
        n_perct = math.ceil((5 * num_nodes) / 100)
        k_hop = math.floor(dim / n_perct)    
        designated_nodes = []
    
        for node in self.sorted_degree_centrality.keys():
            valid_node = True
            for marked_node in designated_nodes:
                if nx.shortest_path_length(self.view.topology(), marked_node, node) < k_hop:
                   valid_node = False
                   break
            if valid_node:
               designated_nodes.append(node)

            if len(designated_nodes) == n_perct:
               break

        self.top_designated_nodes = designated_nodes
        assignments = []
        for edge_node in edge_nodes:
            shortest_dist = float('inf')
            assigned_node = None
            for designated_node in designated_nodes:
                dist = nx.shortest_path_length(self.view.topology(), edge_node, designated_node)
                if dist < shortest_dist:
                   shortest_dist = dist
                   assigned_node = designated_node

            assignments.append((edge_node, assigned_node))
        self.assigned_designated_node =  assignments
        
    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, size, priority, log):
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log, priority)
        for node in self.view.topology().nodes():
            if self.view.has_cache(node):
                try:
                    self.Global_Interest_Counter[node][content]+=1
                except:
                        self.Global_Interest_Counter[node][content] = 1
        
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v, size=size, priority=priority):
                    serving_node = v
                    break
            self.controller.get_content(v, size=size, priority=priority)
            serving_node = v 
        loc_leader_nodes_updated = set()
        loc_updated_contents = set()
        
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for hop in range(1, len(path)):
            u = path[hop - 1]
            v = path[hop]
            self.controller.forward_content_hop(u, v, main_path=True, size=size, priority=priority)
            if self.view.has_cache(u) and u!=source and u!=receiver:
                self.designated_leader = []
                for edge_node, designated_node in self.assigned_designated_node:
                    if u == edge_node:
                        self.designated_leader = designated_node
    
                if u in self.cache_size:
                    Total_size =self.cache_size[u]
                
                if path[hop-1] != receiver and path[hop-1]!= source:
                    cache_contents = self.view.cache_dump2(path[hop-1])
                    if cache_contents == (None):
                        continue
            
                    cache_occupancy = len(cache_contents)
                    available_space =  Total_size - cache_occupancy
                
                if u == path[-2]:
                    try:
                        self.Local_Interest_Counter[u][content] += 1
                    except KeyError:
                            self.Local_Interest_Counter[u][content] = 1
                    try:
                        self.Local_Interest_Counter[self.designated_leader][content] += 1
                    except KeyError:
                            self.Local_Interest_Counter[self.designated_leader][content] = 1
                    
                    Local_Popularity_Count = {key: self.Local_Interest_Counter[key] for key in self.Local_Interest_Counter if key == self.designated_leader}
                
                    self.Global_Popularity_Count = {key: self.Global_Interest_Counter[key] for key in self.Global_Interest_Counter  if key == self.designated_leader}
                    if Local_Popularity_Count and self.Global_Popularity_Count:
                        self.PName = (0.875) * Local_Popularity_Count[self.designated_leader][content] + (0.125) * self.Global_Popularity_Count[self.designated_leader][content]
                
                if available_space > 0:
                    self.controller.put_content(u, size=size, priority=priority)

                elif u == path[-2] and not self.view.cache_lookup(path[-2], content) and available_space == 0:
                    if self.PName >= 50:
                        self.controller.put_content(path[-2], size=size, priority=priority)

                elif u != path[-2] and not self.view.cache_lookup(u, content) and available_space == 0:
                        self.controller.put_content(u, size=size, priority=priority)           
        self.controller.end_session()  
