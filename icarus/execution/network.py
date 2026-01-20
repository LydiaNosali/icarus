"""Network Model-View-Controller (MVC)

This module contains classes providing an abstraction of the network shown to
the strategy implementation. The network is modelled using an MVC design
pattern.

A strategy performs actions on the network by calling methods of the
`NetworkController`, that in turns updates  the `NetworkModel` instance that
updates the `NetworkView` instance. The strategy can get updated information
about the network status by calling methods of the `NetworkView` instance.

The `NetworkController` is also responsible to notify a `DataCollectorProxy`
of all relevant events.
"""
import copy
import logging
import math
import random

import networkx as nx
import fnss

from icarus.registry import CACHE_POLICY
from icarus.util import iround, path_links
import pickle
import json
from pathlib import Path

__all__ = ["NetworkModel", "NetworkView", "NetworkController"]

logger = logging.getLogger("orchestration")


def symmetrify_paths(shortest_paths):
    """Make paths symmetric

    Given a dictionary of all-pair shortest paths, it edits shortest paths to
    ensure that all path are symmetric, e.g., path(u,v) = path(v,u)

    Parameters
    ----------
    shortest_paths : dict of dict
        All pairs shortest paths

    Returns
    -------
    shortest_paths : dict of dict
        All pairs shortest paths, with all paths symmetric

    Notes
    -----
    This function modifies the shortest paths dictionary provided
    """
    for u in shortest_paths:
        for v in shortest_paths[u]:
            shortest_paths[u][v] = list(reversed(shortest_paths[v][u]))
    return shortest_paths


class NetworkView:
    """Network view

    This class provides an interface that strategies and data collectors can
    use to know updated information about the status of the network.
    For example the network view provides information about shortest paths,
    characteristics of links and currently cached objects in nodes.
    """

    def __init__(self, model):
        """Constructor

        Parameters
        ----------
        model : NetworkModel
            The network model instance
        """
        if not isinstance(model, NetworkModel):
            raise ValueError(
                "The model argument must be an instance of " "NetworkModel"
            )
        self.model = model

    def content_locations(self, k):
        """Return a set of all current locations of a specific content.

        This include both persistent content sources and temporary caches.

        Parameters
        ----------
        k : any hashable type
            The content identifier

        Returns
        -------
        nodes : set
            A set of all nodes currently storing the given content
        """
        loc = {v for v in self.model.cache if self.model.cache[v].has(k)}
        source = self.content_source(k)
        if source:
            loc.add(source)
        return loc

    def content_source(self, k):
        """Return the node identifier where the content is persistently stored.

        Parameters
        ----------
        k : any hashable type
            The content identifier

        Returns
        -------
        node : any hashable type
            The node persistently storing the given content or None if the
            source is unavailable
        """
        return self.model.content_source.get(k, None)

    def shortest_path(self, s, t):
        """Return the shortest path from *s* to *t*

        Parameters
        ----------
        s : any hashable type
            Origin node
        t : any hashable type
            Destination node

        Returns
        -------
        shortest_path : list
            List of nodes of the shortest path (origin and destination
            included)
        """
        return self.model.shortest_path[s][t]

    def all_pairs_shortest_paths(self):
        """Return all pairs shortest paths

        Return
        ------
        all_pairs_shortest_paths : dict of lists
            Shortest paths between all pairs
        """
        return self.model.shortest_path

    def cluster(self, v):
        """Return cluster to which a node belongs, if any

        Parameters
        ----------
        v : any hashable type
            Node

        Returns
        -------
        cluster : int
            Cluster to which the node belongs, None if the topology is not
            clustered or the node does not belong to any cluster
        """
        if "cluster" in self.model.topology.node[v]:
            return self.model.topology.node[v]["cluster"]
        else:
            return None

    def link_type(self, u, v):
        """Return the type of link *(u, v)*.

        Type can be either *internal* or *external*

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node

        Returns
        -------
        link_type : str
            The link type
        """
        return self.model.link_type[(u, v)]

    def link_delay(self, u, v):
        """Return the delay of link *(u, v)*.

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node

        Returns
        -------
        delay : float
            The link delay
        """
        return self.model.link_delay[(u, v)]

    def topology(self):
        """Return the network topology

        Returns
        -------
        topology : fnss.Topology
            The topology object

        Notes
        -----
        The topology object returned by this method must not be modified by the
        caller. This object can only be modified through the NetworkController.
        Changes to this object will lead to inconsistent network state.
        """
        return self.model.topology

    def cache_nodes(self, size=False):
        """Returns a list of nodes with caching capability

        Parameters
        ----------
        size: bool, opt
            If *True* return dict mapping nodes with size

        Returns
        -------
        cache_nodes : list or dict
            If size parameter is False or not specified, it is a list of nodes
            with caches. Otherwise it is a dict mapping nodes with a cache
            and their size.
        """
        return (
            {v: c.maxlen for v, c in self.model.cache.items()}
            if size
            else list(self.model.cache.keys())
        )
    
    def cache_tiers(self, node):
        """Returns a dictionary mapping each caching node to its list of tiers.

        Returns
        -------
        per_node_tiers : dict
            Dictionary where keys are node IDs and values are lists of tier dicts.
        """
        return self.model.per_node_tiers.get(node, [])
    
    def has_cache(self, node):
        """Check if a node has a content cache.

        Parameters
        ----------
        node : any hashable type
            The node identifier

        Returns
        -------
        has_cache : bool,
            *True* if the node has a cache, *False* otherwise
        """
        return node in self.model.cache

    def cache_lookup(self, node, content):
        """Check if the cache of a node has a content object, without changing
        the internal state of the cache.

        This method is meant to be used by data collectors to calculate
        metrics. It should not be used by strategies to look up for contents
        during the simulation. Instead they should use
        `NetworkController.get_content`

        Parameters
        ----------
        node : any hashable type
            The node identifier
        content : any hashable type
            The content identifier

        Returns
        -------
        has_content : bool
            *True* if the cache of the node has the content, *False* otherwise.
            If the node does not have a cache, return *None*
        """
        if node in self.model.cache:
            return self.model.cache[node].has(content)

    def local_cache_lookup(self, node, content):
        """Check if the local cache of a node has a content object, without
        changing the internal state of the cache.

        The local cache is an area of the cache of a node reserved for
        uncoordinated caching. This is currently used only by hybrid
        hash-routing strategies.

        This method is meant to be used by data collectors to calculate
        metrics. It should not be used by strategies to look up for contents
        during the simulation. Instead they should use
        `NetworkController.get_content_local_cache`.

        Parameters
        ----------
        node : any hashable type
            The node identifier
        content : any hashable type
            The content identifier

        Returns
        -------
        has_content : bool
            *True* if the cache of the node has the content, *False* otherwise.
            If the node does not have a cache, return *None*
        """
        if node in self.model.local_cache:
            return self.model.local_cache[node].has(content)
        else:
            return False

    def cache_dump(self, node, k=None):
        """Returns the dump of the content of a cache in a specific node

        Parameters
        ----------
        node : any hashable type
            The node identifier

        Returns
        -------
        dump : list
            List of contents currently in the cache
        """
        if node in self.model.cache:
            return self.model.cache[node].dump(k)

    def cache_dump2(self, node, k=None):
        """Returns the dump of the content of a cache in a specific node

        Parameters
        ----------
        node : any hashable type
            The node identifier

        Returns
        -------
        dump : list
            List of contents currently in the cache
        """
        if node in self.model.cache:
            return self.model.cache[node].dump2(k)
 
    def node_state(self, node):
        return self.model.cache[node].node_state()
    
    def get_tier_stats(self):
        return self.model.tier_statistics

    def get_tiers_last_access(self, node):
        if node in self.model.cache:
            return self.model.cache[node].get_tiers_last_access()


class NetworkModel:
    """Models the internal state of the network.

    This object should never be edited by strategies directly, but only through
    calls to the network controller.
    """

    def __init__(self, topology, cache_policy, shortest_path=None, avg_content_size=None, **kwargs):
        """Constructor

        Parameters
        ----------
        topology : fnss.Topology
            The topology object
        cache_policy : dict or Tree
            Cache policy descriptor.
        shortest_path : dict of dict, optional
            The all-pair shortest paths of the network
        """

        if not isinstance(topology, fnss.Topology):
            raise ValueError(
                "The topology argument must be an instance of "
                "fnss.Topology or any of its subclasses."
            )

        self.avg_content_size = avg_content_size
        self.topology = topology
        self.content_source = {}
        self.source_node = {}

        self.link_type = nx.get_edge_attributes(topology, "type")
        self.link_delay = fnss.get_delays(topology)

        if not topology.is_directed():
            for (u, v), link_type in list(self.link_type.items()):
                self.link_type[(v, u)] = link_type
            for (u, v), delay in list(self.link_delay.items()):
                self.link_delay[(v, u)] = delay

        self.cache_size = {}
        
        self.node_carbon_intensity = {}
        self.cache = {}
        
        policy_name = cache_policy["name"]
        policy_args = {k: v for k, v in cache_policy.items() if k != "name"}
        base_tiers = cache_policy.get("tiers", [])
        self.per_node_tiers = cache_policy.get("tiers_per_node", {})
        
        saved_state_file = kwargs.get("saved_state_file")
        for node, data in topology.nodes(data=True):
            self.node_carbon_intensity[node] = data.get("carbon_intensity", random.randint(50,900)) / 1000
            stack_name, stack_props = fnss.get_stack(topology, node)

            if stack_name == "router":
                if "cache_size" in stack_props:
                    size = stack_props["cache_size"]
                    self.cache_size[node] = size
                    
                    if node in self.per_node_tiers:
                        tiers_src = self.per_node_tiers[node]
                    else:
                        tiers_src = base_tiers
                        
                    if tiers_src:
                        node_tier_list = []
                        # 1️⃣ Compute raw sizes
                        logger.info(f"size:{size}")
                        floored_sizes = [round(tier["size_factor"] * size) for tier in tiers_src]
                        logger.info(f"floored_sizes:{floored_sizes}")
                        
                        remainder = int(size - sum(floored_sizes))  # remaining units to distribute
                        logger.info(f"remainder:{remainder}")
                        
                        floored_sizes[len(floored_sizes)-1]+=remainder
                        logger.info(f"floored_sizes:{floored_sizes}")
                        
                        # 3️⃣ Build tier list
                        for tier, actual_size in zip(tiers_src, floored_sizes):
                            tier_copy = tier.copy()
                            tier_copy["actual_size"] = actual_size
                            tier_copy["actual_size_bytes"] = actual_size * self.avg_content_size
                            node_tier_list.append(tier_copy)

                        if node_tier_list and node_tier_list[0]["actual_size"] == 0:
                            node_tier_list[0]["actual_size"] = 1
                            node_tier_list[0]["actual_size_bytes"] = 8000
                            for i in range(len(node_tier_list) - 1, 0, -1):
                                if node_tier_list[i]["actual_size"] > 0:
                                    node_tier_list[i]["actual_size"] -= 1
                                    if node_tier_list[i]["actual_size"] == 1:
                                        node_tier_list[i]["actual_size_bytes"] = 8000 
                                    else:
                                        node_tier_list[i]["actual_size_bytes"] -= self.avg_content_size
                                    break

                        node_tier_list = [t for t in node_tier_list if t["actual_size"] > 0]
                        logger.info(f"node:{node}")
                        logger.info(f"node_tier_list:{node_tier_list}")
                        self.per_node_tiers[node] = node_tier_list
                        node_policy_args = {k: v for k, v in policy_args.items() if k != "tiers"}
                        node_policy_args["tiers"] = node_tier_list

                    self.cache[node] = CACHE_POLICY[policy_name](size, **node_policy_args)
            elif stack_name == "source":
                contents = stack_props.get("contents", [])
                self.source_node[node] = contents
                for content in contents:
                    self.content_source[content] = node

        # --- NEW: compute tier statistics ---
        print(f"[✅] cold_start={not bool(saved_state_file)}, saved_state_file:{saved_state_file}")
        logger.info(f"[✅] cold_start={not bool(saved_state_file)}")
        self.per_node_tiers = {
            n: tiers for n, tiers in self.per_node_tiers.items()
            if n in set(self.cache.keys())
        }

        self.tier_statistics = {}
        self.tier_sizes_mb = {}
        
        for node, tiers in self.per_node_tiers.items():
            for tier in tiers:
                tier_name = tier["name"]
                if tier_name not in self.tier_statistics:
                    self.tier_statistics[tier_name] = 0
                self.tier_statistics[tier_name] += 1
                 # Sum sizes in GB
                size_bytes = tier["actual_size_bytes"]
                if tier_name not in self.tier_sizes_mb:
                    self.tier_sizes_mb[tier_name] = 0
                self.tier_sizes_mb[tier_name] += size_bytes / (1024 * 1024)  # convert bytes -> MB

        # print(f"Tier statistics: {self.tier_statistics}")
        # print(f"Tier sizes (MB): {self.tier_sizes_mb}")

        # Local uncoordinated cache (for edge cache mode)
        self.local_cache = {}

        # Failure simulation state
        self.removed_nodes = {}
        self.disconnected_neighbors = {}
        self.removed_links = {}
        self.removed_sources = {}
        self.removed_caches = {}
        self.removed_local_caches = {}

        # ==========================================================
        # ✅ AUTOLOAD PREVIOUS STATE (optional)
        # ==========================================================
        if saved_state_file :
            saved_dir = Path("network_states")
            # Choose filename based on period number if provided
            saved_state_path = saved_dir / f"{saved_state_file}.pkl"
            Path(saved_state_path).exists()
            try:
                print(f"[♻️] Loading saved network state from {saved_state_path} ...")
                with open(saved_state_path, "rb") as f:
                    state = pickle.load(f)

                # Restore per-node tiers and capacities
                if "old_per_node_tiers" in state:
                    new_tiers = self.rebuild_tiers(state["old_per_node_tiers"], self.cache_size)
                    for node, cache_state in new_tiers.items():
                        if node in self.cache and hasattr(self.cache[node], "restore_from_dump"):
                            self.cache[node].restore_from_dump(cache_state)
                print(f"[✅] NetworkModel restored from {saved_state_path}")
            except Exception as e:
                print(f"[⚠️] Failed to load saved network state: {e}")

        # Shortest paths of the network
        self.shortest_path = (
            dict(shortest_path)
            if shortest_path is not None
            else symmetrify_paths(dict(nx.all_pairs_dijkstra_path(topology)))
        )
        # node_greenness = topology.graph.get("node_greenness", {})
        # if node_greenness  == {}:
        #     # print(f"USING LATENCIES")
        #     # self.shortest_path = (
        #     #         dict(shortest_path)
        #     #         if shortest_path is not None
        #     #         else self.build_carbon_aware_paths(topology, self.node_carbon_intensity)
        #     #     )
        #     self.shortest_path = (
        #     dict(shortest_path)
        #     if shortest_path is not None
        #     else symmetrify_paths(dict(nx.all_pairs_dijkstra_path(topology)))
        #     )
        # else:
        #     # print(f"USING TOPSIS SCORES")
        #     self.shortest_path = (
        #         dict(shortest_path)
        #         if shortest_path is not None
        #         else self.build_carbon_aware_paths(topology, node_greenness)
        #     )

    def build_carbon_aware_paths(self, topology, node_greenness):
        def weight(u, v, data):
            return node_greenness.get(v, 1.0)
        return symmetrify_paths(
            dict(nx.all_pairs_dijkstra_path(topology, weight=weight))
        )

    def rebuild_tiers(self, old_cache_tiers_per_node, new_cache_sizes):
        new_per_node_tiers = {}

        # Helper for consistent eviction from a tier
        def _evict_from_tier(node_state, tinfo, qname):
            q = tinfo[qname]
            if not q:
                return None

            removed = q.pop(0)  # LRU (right side)

            # Remove from global ARC queues
            for gq in ["t1", "t2", "b1", "b2"]:
                try:
                    node_state["global"][gq].remove(removed)
                except ValueError:
                    pass

            # Remove from global cache dict
            node_state["global"]["_cache"].pop(removed, None)

            tinfo[qname] = q
            return removed

        for node, node_state in old_cache_tiers_per_node.items():
            logger.info(f"node:{node}")
            try:
                if node not in new_cache_sizes:
                    continue 
                
                total_new = new_cache_sizes[node]

                # Sort tiers fastest → slowest by latency
                tier_configs = sorted(
                    self.per_node_tiers[node],
                    key=lambda t: float(t.get("latency", 1.0)),
                )
                tiers_sorted = [
                    t["name"] for t in tier_configs
                    if t["name"] in node_state["tiers"]
                ]

                if not tiers_sorted:
                    new_per_node_tiers[node] = node_state
                    continue

                # ---- 1) Compute and apply new maxlen for ALL tiers on this node ----
                new_maxlens = {}
                for tier in tier_configs:
                    name = tier["name"]
                    if name not in node_state["tiers"]:
                        continue
                    new_maxlens[name] = tier["actual_size"]
                
                for name, length in new_maxlens.items():
                    node_state["tiers"][name]["maxlen"] = int(length)

                # ---- 2) Build current sizes and targets ----
                curr = {}
                tgt = {}
                for name in tiers_sorted:
                    tinfo = node_state["tiers"][name]
                    curr[name] = len(tinfo["t1"]) + len(tinfo["t2"])
                    tgt[name] = new_maxlens[name]
                
                logger.info(f"curr:{curr}")
                logger.info(f"new_maxlens:{new_maxlens}")
                
                # ---- 3) Strict cascade: fast → immediate slower tier ----
                for i, fast_name in enumerate(tiers_sorted[:-1]):
                    t_fast = node_state["tiers"][fast_name]

                    while curr[fast_name] > tgt[fast_name] and (t_fast["t1"] or t_fast["t2"]):
                        src_qname = "t2" if t_fast["t2"] else "t1"
                        k = t_fast[src_qname].pop(0)  # LRU from fast tier
                        curr[fast_name] -= 1

                        slow_name = tiers_sorted[i + 1]
                        t_slow = node_state["tiers"][slow_name]
                        dst_qname = src_qname
                        t_slow[dst_qname].append(k) # MRU in slow tier
                        curr[slow_name] += 1

                # ---- 4) Evict ONLY from the slowest tier if still over capacity ----
                slowest = tiers_sorted[-1]
                tinfo = node_state["tiers"][slowest]
                over_last = max(0, curr[slowest] - tgt[slowest])

                while over_last > 0 and (tinfo["t2"] or tinfo["t1"]):
                    if tinfo["t2"]:
                        _evict_from_tier(node_state, tinfo, "t2")
                    else:
                        _evict_from_tier(node_state, tinfo, "t1")
                    curr[slowest] -= 1
                    over_last -= 1

                # ---- 5) Safety guard: per-tier enforcement (rounding, etc.) ----
                for name in tiers_sorted:
                    tinfo = node_state["tiers"][name]
                    maxlen = tgt[name]
                    size_now = len(tinfo["t1"]) + len(tinfo["t2"])
                    extra = max(0, size_now - maxlen)

                    while extra > 0 and (tinfo["t2"] or tinfo["t1"]):
                        if tinfo["t2"]:
                            _evict_from_tier(node_state, tinfo, "t2")
                        else:
                            _evict_from_tier(node_state, tinfo, "t1")
                        curr[name] -= 1
                        extra -= 1
                
                # ---- 6) Enforce |B1| + |B2| ≤ C ----

                B1 = node_state["global"]["b1"]
                B2 = node_state["global"]["b2"]
                B1.clear()
                B2.clear()

                # ---- 7) Clamp p ----
                node_state["global"]["p"] = min(
                    node_state["global"]["p"], total_new
                )
                # ---- 8) Consistency check between tiers and global ----
                global_t1 = node_state["global"]["t1"]
                global_t2 = node_state["global"]["t2"]
                global_cache = node_state["global"]["_cache"]

                # 8a) collect all keys that appear in any tier queue
                tier_keys = set()
                for tname in tiers_sorted:
                    tinfo = node_state["tiers"][tname]
                    tier_keys.update(tinfo["t1"])
                    tier_keys.update(tinfo["t2"])

                # 8b) ensure global T1/T2 only contain keys that are still in tiers
                for qname, gq in (("t1", global_t1), ("t2", global_t2)):
                    to_remove = [k for k in gq if k not in tier_keys]
                    # if to_remove:
                    #     logger.warning(
                    #         f"[rebuild_tiers] node {node}: removing {len(to_remove)} "
                    #         f"orphan(s) from global {qname}: {to_remove}"
                    #     )
                    for k in to_remove:
                        gq.remove(k)
                        global_cache.pop(k, None)
                
                # 8c) ensure all tier keys appear in global T1/T2/_cache
                global_keys = set(global_t1) | set(global_t2)
                missing = tier_keys - global_keys
                if missing:
                    logger.warning(
                        f"[rebuild_tiers] node {node}: {len(missing)} keys in tiers "
                        f"missing from global T1/T2: {missing}"
                    )
                    # Option 1: re-insert them into T1 and _cache
                    for k in missing:
                        if k not in global_cache:
                            # you may want to reconstruct the value differently
                            global_cache[k] = None
                        if k not in global_t1 and k not in global_t2:
                            global_t1.append(k)  # or T2, depending on your policy

                new_per_node_tiers[node] = node_state

            except Exception as e:
                print(f"[⚠️] Failed to rebuild tiers for node {node}: {e}")

        return new_per_node_tiers
    
    def save_state(self, filename_prefix="network_state", directory="network_states"):
        Path(directory).mkdir(exist_ok=True)
        filepath = Path(directory) / f"{filename_prefix}.pkl"
        
        old_cache_sizes = copy.deepcopy(self.cache_size)
        old_per_node_tiers = {node: cache.node_state() for node, cache in self.cache.items()}

        collector_proxy = getattr(self, "collector_proxy", None)
        network_cache = getattr(self, "network_cache", None)
        cache_placement = getattr(self, "cache_placement", None)
        alpha = getattr(self, "alpha", None)
        root = collector_proxy.results()

        state = {
            "network_cache": network_cache,
            "cache_placement":cache_placement,
            "alpha":alpha,
            "old_cache_size": old_cache_sizes,
            "old_cache_size_total": sum(old_cache_sizes.values()),
            "old_tier_statistics": self.tier_statistics,
            "old_tier_sizes_mb": self.tier_sizes_mb,
            "old_per_node_tiers": old_per_node_tiers,
            "old_results":root,
            "old_ci":self.node_carbon_intensity,
        }

        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        print(f"[💾] Saved network model state to {filepath}")
        
        jsonpath = filepath.with_suffix(".json")
        with open(jsonpath, "w") as jf:
            json.dump(state, jf, indent=2)
        # print(f"[📄] JSON copy saved to {jsonpath}")
        return str(filepath)


class NetworkController:
    """Network controller

    This class is in charge of executing operations on the network model on
    behalf of a strategy implementation. It is also in charge of notifying
    data collectors of relevant events.
    """

    def __init__(self, model):
        """Constructor

        Parameters
        ----------
        model : NetworkModel
            Instance of the network model
        """
        self.session = None
        self.model = model
        self.collector = None

    def attach_collector(self, collector):
        """Attach a data collector to which all events will be reported.

        Parameters
        ----------
        collector : DataCollector
            The data collector
        """
        self.collector = collector

    def detach_collector(self):
        """Detach the data collector."""
        self.collector = None

    def start_session(self, timestamp, receiver, content, log, priority):
        """Instruct the controller to start a new session (i.e. the retrieval
        of a content).

        Parameters
        ----------
        timestamp : int
            The timestamp of the event
        receiver : any hashable type
            The receiver node requesting a content
        content : any hashable type
            The content identifier requested by the receiver
        log : bool
            *True* if this session needs to be reported to the collector,
            *False* otherwise
        """
        self.session = dict(
            timestamp=timestamp, receiver=receiver, content=content, log=log, priority=priority
        )
        if self.collector is not None and self.session["log"]:
            self.collector.start_session(timestamp, receiver, content, priority)

    def forward_request_path(self, s, t, **kwargs):
        """Forward a request from node *s* to node *t* over the provided path.

        Parameters
        ----------
        s : any hashable type
            Origin node
        t : any hashable type
            Destination node
        path : list, optional
            The path to use. If not provided, shortest path is used
        main_path : bool, optional
            If *True*, indicates that link path is on the main path that will
            lead to hit a content. It is normally used to calculate latency
            correctly in multicast cases. Default value is *True*
        """
        path = kwargs["path"] or None
        if path is None:
            path = self.model.shortest_path[s][t]
        for u, v in path_links(path):
            self.forward_request_hop(u, v, **kwargs)

    def forward_content_path(self, u, v, **kwargs):
        """Forward a content from node *s* to node *t* over the provided path.

        Parameters
        ----------
        s : any hashable type
            Origin node
        t : any hashable type
            Destination node
        path : list, optional
            The path to use. If not provided, shortest path is used
        main_path : bool, optional
            If *True*, indicates that this path is being traversed by content
            that will be delivered to the receiver. This is needed to
            calculate latency correctly in multicast cases. Default value is
            *True*
        """
        path = kwargs["path"] or None
        if path is None:
            path = self.model.shortest_path[u][v]
        for u, v in path_links(path):
            self.forward_content_hop(u, v, **kwargs)

    def forward_request_hop(self, u, v, **kwargs):
        """Forward a request over link  u -> v.

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node
        main_path : bool, optional
            If *True*, indicates that link link is on the main path that will
            lead to hit a content. It is normally used to calculate latency
            correctly in multicast cases. Default value is *True*
        """
        main_path = kwargs.get("main_path") or True
        if self.collector is not None and self.session["log"]:
            self.collector.request_hop(u, v, main_path=main_path, carbon_intensity=self.model.node_carbon_intensity[v])

    def forward_content_hop(self, u, v, **kwargs):
        """Forward a content over link  u -> v.

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node
        main_path : bool, optional
            If *True*, indicates that this link is being traversed by content
            that will be delivered to the receiver. This is needed to
            calculate latency correctly in multicast cases. Default value is
            *True*
        """
        if self.collector is not None and self.session["log"]:
            self.collector.content_hop(u, v, carbon_intensity=self.model.node_carbon_intensity[v], **kwargs)

    def put_content(self, node, **kwargs):
        """Store content in the specified node.

        The node must have a cache stack and the actual insertion of the
        content is executed according to the caching policy. If the caching
        policy has a selective insertion policy, then content may not be
        inserted.

        Parameters
        ----------
        node : any hashable type
            The node where the content is inserted

        Returns
        -------
        evicted : any hashable type
            The evicted object or *None* if no contents were evicted.
        """
        if node in self.model.cache:
            content = self.session["content"]
            logger.info(f"put content: {content} in cache {node}")
            res = self.model.cache[node].put(self.session["content"], self.session["priority"], **kwargs)
            if (res is None or type(res) is int) and self.collector is not None and self.session["log"]:
                self.collector.write_content(node, cache_tiers=self.model.per_node_tiers[node], carbon_intensity=self.model.node_carbon_intensity[node],**kwargs)
            return res 

    def get_content(self, node, **kwargs):
        """Get a content from a server or a cache.

        Parameters
        ----------
        node : any hashable type
            The node where the content is retrieved

        Returns
        -------
        content : bool
            True if the content is available, False otherwise
        """
        content = self.session["content"]
        if node in self.model.cache:
            logger.info(f"is content:{content} in cache {node}")
            cache_hit = self.model.cache[node].get(self.session["content"], self.session["priority"])
            logger.info(f"{cache_hit}")
            if cache_hit:
                if self.session["log"]:
                    tier_index = self.get_tier_index(node, self.session["content"], self.session['priority'])
                    self.collector.cache_hit(node, cache_tiers=self.model.per_node_tiers[node], tier_index=tier_index, carbon_intensity=self.model.node_carbon_intensity[node], **kwargs)
            else:
                if self.session["log"]:
                    self.collector.cache_miss(node)
            return cache_hit
        name, props = fnss.get_stack(self.model.topology, node)
        if name == "source" and self.session["content"] in props["contents"]:
            if self.collector is not None and self.session["log"]:
                logger.info(f"content:{content} in server {node}")
                self.collector.server_hit(node, server_size=len(self.model.source_node[node]), carbon_intensity=self.model.node_carbon_intensity[node], **kwargs)
            return True
        else:
            return False

    def end_session(self, success=True):
        """Close a session

        Parameters
        ----------
        success : bool, optional
            *True* if the session was completed successfully, *False* otherwise
        """
        if self.collector is not None and self.session["log"]:
            self.collector.end_session(success)
        self.session = None

    def rewire_link(self, u, v, up, vp, recompute_paths=True):
        """Rewire an existing link to new endpoints

        This method can be used to model mobility patters, e.g., changing
        attachment points of sources and/or receivers.

        Note well. With great power comes great responsibility. Be careful when
        using this method. In fact as a result of link rewiring, network
        partitions and other corner cases might occur. Ensure that the
        implementation of strategies using this method deal with all potential
        corner cases appropriately.

        Parameters
        ----------
        u, v : any hashable type
            Endpoints of link before rewiring
        up, vp : any hashable type
            Endpoints of link after rewiring
        """
        link = self.model.topology.adj[u][v]
        self.model.topology.remove_edge(u, v)
        self.model.topology.add_edge(up, vp, **link)
        if recompute_paths:
            shortest_path = dict(nx.all_pairs_dijkstra_path(self.model.topology))
            self.model.shortest_path = symmetrify_paths(shortest_path)

    def remove_link(self, u, v, recompute_paths=True):
        """Remove a link from the topology and update the network model.

        Note well. With great power comes great responsibility. Be careful when
        using this method. In fact as a result of link removal, network
        partitions and other corner cases might occur. Ensure that the
        implementation of strategies using this method deal with all potential
        corner cases appropriately.

        Also, note that, for these changes to be effective, the strategy must
        use fresh data provided by the network view and not storing local copies
        of network state because they won't be updated by this method.

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node
        recompute_paths: bool, optional
            If True, recompute all shortest paths
        """
        self.model.removed_links[(u, v)] = self.model.topology.adj[u][v]
        self.model.topology.remove_edge(u, v)
        if recompute_paths:
            shortest_path = dict(nx.all_pairs_dijkstra_path(self.model.topology))
            self.model.shortest_path = symmetrify_paths(shortest_path)

    def restore_link(self, u, v, recompute_paths=True):
        """Restore a previously-removed link and update the network model

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node
        recompute_paths: bool, optional
            If True, recompute all shortest paths
        """
        self.model.topology.add_edge(u, v, **self.model.removed_links.pop((u, v)))
        if recompute_paths:
            shortest_path = dict(nx.all_pairs_dijkstra_path(self.model.topology))
            self.model.shortest_path = symmetrify_paths(shortest_path)

    def remove_node(self, v, recompute_paths=True):
        """Remove a node from the topology and update the network model.

        Note well. With great power comes great responsibility. Be careful when
        using this method. In fact, as a result of node removal, network
        partitions and other corner cases might occur. Ensure that the
        implementation of strategies using this method deal with all potential
        corner cases appropriately.

        It should be noted that when this method is called, all links connected
        to the node to be removed are removed as well. These links are however
        restored when the node is restored. However, if a link attached to this
        node was previously removed using the remove_link method, restoring the
        node won't restore that link as well. It will need to be restored with a
        call to restore_link.

        This method is normally quite safe when applied to remove cache nodes or
        routers if this does not cause partitions. If used to remove content
        sources or receiver, special attention is required. In particular, if
        a source is removed, the content items stored by that source will no
        longer be available if not cached elsewhere.

        Also, note that, for these changes to be effective, the strategy must
        use fresh data provided by the network view and not storing local copies
        of network state because they won't be updated by this method.

        Parameters
        ----------
        v : any hashable type
            Node to remove
        recompute_paths: bool, optional
            If True, recompute all shortest paths
        """
        self.model.removed_nodes[v] = self.model.topology.node[v]
        # First need to remove all links the removed node as endpoint
        neighbors = self.model.topology.adj[v]
        self.model.disconnected_neighbors[v] = set(neighbors.keys())
        for u in self.model.disconnected_neighbors[v]:
            self.remove_link(v, u, recompute_paths=False)
        self.model.topology.remove_node(v)
        if v in self.model.cache:
            self.model.removed_caches[v] = self.model.cache.pop(v)
        if v in self.model.local_cache:
            self.model.removed_local_caches[v] = self.model.local_cache.pop(v)
        if v in self.model.source_node:
            self.model.removed_sources[v] = self.model.source_node.pop(v)
            for content in self.model.removed_sources[v]:
                self.model.countent_source.pop(content)
        if recompute_paths:
            shortest_path = dict(nx.all_pairs_dijkstra_path(self.model.topology))
            self.model.shortest_path = symmetrify_paths(shortest_path)

    def restore_node(self, v, recompute_paths=True):
        """Restore a previously-removed node and update the network model.

        Parameters
        ----------
        v : any hashable type
            Node to restore
        recompute_paths: bool, optional
            If True, recompute all shortest paths
        """
        self.model.topology.add_node(v, **self.model.removed_nodes.pop(v))
        for u in self.model.disconnected_neighbors[v]:
            if (v, u) in self.model.removed_links:
                self.restore_link(v, u, recompute_paths=False)
        self.model.disconnected_neighbors.pop(v)
        if v in self.model.removed_caches:
            self.model.cache[v] = self.model.removed_caches.pop(v)
        if v in self.model.removed_local_caches:
            self.model.local_cache[v] = self.model.removed_local_caches.pop(v)
        if v in self.model.removed_sources:
            self.model.source_node[v] = self.model.removed_sources.pop(v)
            for content in self.model.source_node[v]:
                self.model.countent_source[content] = v
        if recompute_paths:
            shortest_path = dict(nx.all_pairs_dijkstra_path(self.model.topology))
            self.model.shortest_path = symmetrify_paths(shortest_path)

    def reserve_local_cache(self, ratio=0.1):
        """Reserve a fraction of cache as local.

        This method reserves a fixed fraction of the cache of each caching node
        to act as local uncoodinated cache. Methods `get_content` and
        `put_content` will only operated to the coordinated cache. The reserved
        local cache can be accessed with methods `get_content_local_cache` and
        `put_content_local_cache`.

        This function is currently used only by hybrid hash-routing strategies.

        Parameters
        ----------
        ratio : float
            The ratio of cache space to be reserved as local cache.
        """
        if ratio < 0 or ratio > 1:
            raise ValueError("ratio must be between 0 and 1")
        for v, c in list(self.model.cache.items()):
            maxlen = iround(c.maxlen * (1 - ratio))
            if maxlen > 0:
                self.model.cache[v] = type(c)(maxlen)
            else:
                # If the coordinated cache size is zero, then remove cache
                # from that location
                if v in self.model.cache:
                    self.model.cache.pop(v)
            local_maxlen = iround(c.maxlen * (ratio))
            if local_maxlen > 0:
                self.model.local_cache[v] = type(c)(local_maxlen)

    def get_content_local_cache(self, node):
        """Get content from local cache of node (if any)

        Get content from a local cache of a node. Local cache must be
        initialized with the `reserve_local_cache` method.

        Parameters
        ----------
        node : any hashable type
            The node to query
        """
        if node not in self.model.local_cache:
            return False
        cache_hit = self.model.local_cache[node].get(self.session["content"], self.session["priority"])
        if cache_hit:
            if self.session["log"]:
                self.collector.cache_hit(node)
        else:
            if self.session["log"]:
                self.collector.cache_miss(node)
        return cache_hit

    def put_content_local_cache(self, node):
        """Put content into local cache of node (if any)

        Put content into a local cache of a node. Local cache must be
        initialized with the `reserve_local_cache` method.

        Parameters
        ----------
        node : any hashable type
            The node to query
        """
        if node in self.model.local_cache:
            return self.model.local_cache[node].put(self.session["content"], self.session["priority"])

    def get_tier_index(self, node, content, priority):
        if node in self.model.cache:
            return self.model.cache[node].get_tier_index(content, priority)
        
    def storage_div(self, path, storage_gain, storage_loss, dep, stor, band, trans, pen, min_gain):
        if self.collector is not None and self.session["log"]:
            self.collector.storage_div(path, storage_gain, storage_loss, dep, stor, band, trans, pen, min_gain)
    