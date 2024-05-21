"""Implementations of all on-path strategies"""
import random

import networkx as nx

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
    "CostCache",
]


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
    def process_event(self, time, receiver, content, log):
        source = self.view.content_source(content)
        self.controller.start_session(time, receiver, content, log)
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
    def process_event(self, time, receiver, content, log, priority):
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
                if self.controller.get_content(v):
                    serving_node = v
                else:
                    # Cache miss, get content from source
                    self.controller.forward_request_path(v, source)
                    self.controller.get_content(source)
                    serving_node = source
                break
        else:
            # No caches on the path at all, get it from source
            self.controller.get_content(v)
            serving_node = v

        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        self.controller.forward_content_path(serving_node, receiver, path)
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
    def process_event(self, time, receiver, content, log, priority):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log, priority)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                # insert content
                self.controller.put_content(v)
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
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        # Leave a copy of the content only in the cache one level down the hit
        # caching node
        copied = False
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if not copied and v != receiver and self.view.has_cache(v):
                self.controller.put_content(v)
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
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for hop in range(1, len(path)):
            u = path[hop - 1]
            v = path[hop]
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
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
            self.controller.forward_content_hop(u, v)
            if v != receiver and v in self.cache_size:
                # The (x/c) factor raised to the power of "c" according to the
                # extended version of ProbCache published in IEEE TPDS
                prob_cache = float(N) / (self.t_tw * self.cache_size[v]) * (x / c) ** c
                if random.random() < prob_cache:
                    self.controller.put_content(v)
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
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        # No cache hits, get content from source
        else:
            self.controller.get_content(v)
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
            self.controller.forward_content_hop(u, v)
            if v == designated_cache:
                self.controller.put_content(v)
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
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if v != receiver and self.view.has_cache(v):
                if random.random() < self.p:
                    self.controller.put_content(v)
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
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        caches = [v for v in path[1:-1] if self.view.has_cache(v)]
        designated_cache = random.choice(caches) if len(caches) > 0 else None
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if v == designated_cache:
                self.controller.put_content(v)
        self.controller.end_session()


@register_strategy("COST_CACHE")
class CostCache(Strategy):
    """CostCache strategy 

    This strategy caches content objects based on a cost function.
    If  storage gain (cost of retrieval from closest node) 
        > 
        storage loss (cost of storage + min (storage gain 
        of data to be evicted from node)) --> cache data
    else --> don't cache
    The cost function includes:
        storage cost: occupation + energy + QoS penalty
        retrieval cost: throughput + energy + QoS penalty
    
    Since the nodes are multi-tier, the device with the worst metrics will be chosen 
    QM-ARC will perhaps use the energy of each device to decide where to store the data.
    
    Storage cost at node:
        occupation cost = (sum_of_content_size+size_content)/cache_size*amz_cost_j)
        energy cost = size_content*un_en_cost
        penalty cost = penalty(retieval_time_des) (perhaps use collector Latency)
    Retrieval cost from des:
        throughput cost = sum_on_nodes_j_src_to_des(size_content/throughput_j*amz_cost_j)
        energy cost : links + nodes
            links  = hop_count_src_des*size_content*un_en_cost_link
            nodes = hop_count_src_des+1*size_content*un_en_cost_node
        penalty cost = penalty(retieval_time_des) (perhaps use collector Latency)
    

    Storage gain == cost of retrieval from closest node des:
    Storage loss == storage cost + min (storage gain of content to evict)
         
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super().__init__(view, controller)
        self.cache_size = view.cache_nodes(size=True)
        # in US: 0.165 $/kWh
        # 0.020324 $/joule
        # caching power density 10^-9 w/bit
        # link energy density j/bit
        # router energy density 2x10^-8 j/bit
        # total_energy_consumption = (caching_power_density * duration_in_seconds + link_energy_density + router_energy_density) * amount_of_data
        # cost_in_dollars = total_energy_consumption * cost_per_joule
        self.content_size = 1500
        self.node_energy_unitary_cost = 1.2
        self.link_energy_unitary_cost = 0.2
        self.storage_energy_unitary_cost = 0.00000144 # $/bits
    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log, priority):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        cached_data = self.view.cache_dump(receiver)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log, priority)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        
        # Return content
        paths = {}
        min_value = 0
        if cached_data:
            for content in cached_data:
                serving_node = self.get_serving_node(receiver, content)
                path = list(reversed(self.view.shortest_path(receiver, serving_node)))
                paths[content] = self.calculate_storage_gain(path)
            min_value = min(paths.values())
        
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))

        storage_gain = self.calculate_storage_gain(path)
        storage_loss = self.calculate_storage_loss(receiver, min_value)

        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                if storage_gain > storage_loss:
                    # insert content
                    self.controller.put_content(v)
        self.controller.end_session()
    

    def get_serving_node(self, receiver, content):
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        for hop in path_links(path):
            if self.view.has_cache(hop):
                if self.controller.get_content(hop):
                    serving_node = hop
                    break
            serving_node = hop
        return serving_node

    def calculate_storage_gain(self, path):
        return self.calculate_throughput_cost(path) + self.calculate_transmission_energy_cost(path)
    
    def calculate_storage_loss(self, receiver, min_value):
        return self.calculate_occupation_cost(receiver) + self.calculate_storage_energy_cost() + min_value

    def calculate_occupation_cost(self, receiver):
        current_cache_size = 0
        amz_cost = 0.00479119 # in $ cost model paper 2016
        if self.view.cache_dump(receiver):
            current_cache_size = len(self.view.cache_dump(receiver))
        if receiver in self.cache_size:
            cache_maxlen = self.cache_size[receiver]
            occupation_cost = (current_cache_size + 1) / cache_maxlen * amz_cost
        else:
            occupation_cost = 0
        return occupation_cost
    
    def calculate_storage_energy_cost(self):
        # TODO store the unitary cost and take the worst one 
        storage_energy_cost = self.content_size  * self.storage_energy_unitary_cost
        return storage_energy_cost

    def calculate_throughput_cost(self, path):
        # TODO store the throughput and take the worst one 
        throughput_cost = 0
        throughput = 1.0
        amz_cost = 0.00479119 # in $ cost model paper 2016
        throughput_cost = (self.content_size / throughput) * amz_cost * len(path)
        return throughput_cost
    
    def calculate_transmission_energy_cost(self, path):
        # TODO store the unitary cost and take the worst one
        transmission_energy_cost = len(path)*self.content_size*self.link_energy_unitary_cost \
                                    + (len(path) + 1)*self.content_size*self.node_energy_unitary_cost 
        return transmission_energy_cost