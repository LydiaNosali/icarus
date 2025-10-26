"""Cache placement strategies

This module provides algorithms for performing cache placement, i.e., given
a cumulative cache size and a topology where each possible node candidate is
labelled, these functions deploy caching space to the nodes of the topology.
"""
import random
import networkx as nx
import numpy as np

from icarus.util import iround
from icarus.registry import register_cache_placement
from icarus.scenarios.algorithms import (
    compute_clusters,
    compute_p_median,
    deploy_clusters,
)

__all__ = [
    "uniform_cache_placement",
    "degree_centrality_cache_placement",
    "betweenness_centrality_cache_placement",
    "uniform_consolidated_cache_placement",
    "random_cache_placement",
    "optimal_median_cache_placement",
    "optimal_hashrouting_cache_placement",
    "clustered_hashrouting_cache_placement",
    "green_cache_placement",
    "hybrid_cache_placement",
    "allocated_cache_placement"
]


@register_cache_placement("UNIFORM")
def uniform_cache_placement(topology, cache_budget, **kwargs):
    """Places cache budget uniformly across cache nodes.

    Parameters
    ----------
    topology : Topology
        The topology object
    cache_budget : int
        The cumulative cache budget
    """
    icr_candidates = topology.graph["icr_candidates"]
    cache_size = iround(cache_budget / len(icr_candidates))
    for v in icr_candidates:
        topology.node[v]["stack"][1]["cache_size"] = cache_size


@register_cache_placement("DEGREE")
def degree_centrality_cache_placement(topology, cache_budget, **kwargs):
    """Places cache budget proportionally to the degree of the node.

    Parameters
    ----------
    topology : Topology
        The topology object
    cache_budget : int
        The cumulative cache budget
    """
    deg = dict(nx.degree(topology))
    icr_candidates = set(topology.graph["icr_candidates"])
    total_deg = sum(v for k, v in deg.items() if k in icr_candidates)
    for v in icr_candidates:
        topology.node[v]["stack"][1]["cache_size"] = iround(
            cache_budget * deg[v] / total_deg
        )


@register_cache_placement("BETWEENNESS_CENTRALITY")
def betweenness_centrality_cache_placement(topology, cache_budget, **kwargs):
    """Assigns cache only to nodes with non-zero betweenness-based allocation.
    
    Ensures Icarus doesn't see cache_size=0 on any node.
    """
    # Step 1: Compute betweenness centrality
    betw = dict(nx.betweenness_centrality(topology))
    icr_candidates = list(topology.graph["icr_candidates"])
    centralities = {v: betw[v] for v in icr_candidates}

    # Step 2: Remove candidates with 0 centrality (they shouldn't get any cache)
    centralities = {v: c for v, c in centralities.items() if c > 0}
    if not centralities:
        return

    # Step 3: Allocate cache proportionally
    total_centrality = sum(centralities.values())
    raw_alloc = {
        v: cache_budget * centralities[v] / total_centrality for v in centralities
    }

    # Step 4: Round allocation, enforce ≥1, and adjust total
    rounded_alloc = {v: max(1, round(raw_alloc[v])) for v in raw_alloc}
    total_allocated = sum(rounded_alloc.values())

    # Adjust if we over-allocated due to enforcing ≥1
    while total_allocated > cache_budget:
        # Find the node with the smallest allocation > 1 to reduce
        over_nodes = [v for v in rounded_alloc if rounded_alloc[v] > 1]
        if not over_nodes:
            break  # Can't reduce anymore without violating ≥1 constraint
        # Reduce the one with the smallest centrality
        victim = min(over_nodes, key=lambda v: centralities[v])
        rounded_alloc[victim] -= 1
        total_allocated -= 1

    # Step 5: Apply allocation
    for v in icr_candidates:
        if v in rounded_alloc:
            topology.node[v]["stack"][1]["cache_size"] = rounded_alloc[v]
        else:
            # Unused ICR candidate – mark explicitly non-caching if needed
            if "cache_size" in topology.node[v]["stack"][1]:
                del topology.node[v]["stack"][1]["cache_size"]


@register_cache_placement("CONSOLIDATED")
def uniform_consolidated_cache_placement(
    topology, cache_budget, spread=0.5, metric_dict=None, target="top", **kwargs
):
    """Consolidate caches in nodes with top centrality.

    Differently from other cache placement strategies that place cache space
    to all nodes but proportionally to their centrality, this strategy places
    caches of all the same size in a set of selected nodes.

    Parameters
    ----------
    topology : Topology
        The topology object
    cache_budget : int
        The cumulative cache budget
    spread : float [0, 1], optional
        The spread factor, The greater it is the more the cache budget is
        spread among nodes. If it is 1, all candidate nodes are assigned a
        cache, if it is 0, only the node with the highest/lowest centrality
        is assigned a cache
    metric_dict : dict, optional
        The centrality metric according to which nodes are selected. If not
        specified, betweenness centrality is selected.
    target : ("top" | "bottom"), optional
        The subsection of the ranked node on which to the deploy caches.
    """
    if spread < 0 or spread > 1:
        raise ValueError("spread factor must be between 0 and 1")
    if target not in ("top", "bottom"):
        raise ValueError('target argument must be either "top" or "bottom"')
    if metric_dict is None and spread < 1:
        metric_dict = nx.betweenness_centrality(topology)

    icr_candidates = topology.graph["icr_candidates"]
    if spread == 1:
        target_nodes = icr_candidates
    else:
        nodes = sorted(icr_candidates, key=lambda k: metric_dict[k])
        if target == "top":
            nodes = list(reversed(nodes))
        # cutoff node must be at least one otherwise, if spread is too low, no
        # nodes would be selected
        cutoff = max(1, iround(spread * len(nodes)))
        target_nodes = nodes[:cutoff]
    cache_size = iround(cache_budget / len(target_nodes))
    if cache_size == 0:
        return
    for v in target_nodes:
        topology.node[v]["stack"][1]["cache_size"] = cache_size


@register_cache_placement("RANDOM")
def random_cache_placement(topology, cache_budget, n_cache_nodes, seed=None, **kwargs):
    """Deploy caching nodes randomly

    Parameters
    ----------
    topology : Topology
        The topology object
    cache_budget : int
        The cumulative cache budget
    n_nodes : int
        The number of caching nodes to deploy
    """
    n_cache_nodes = int(n_cache_nodes)
    icr_candidates = topology.graph["icr_candidates"]
    if len(icr_candidates) < n_cache_nodes:
        raise ValueError(
            "The number of ICR candidates is lower than the target number of caches"
        )
    elif len(icr_candidates) == n_cache_nodes:
        caches = icr_candidates
    else:
        random.seed(seed)
        caches = random.sample(icr_candidates, n_cache_nodes)
    cache_size = iround(cache_budget / n_cache_nodes)
    if cache_size == 0:
        return
    for v in caches:
        topology.node[v]["stack"][1]["cache_size"] = cache_size


@register_cache_placement("OPTIMAL_MEDIAN")
def optimal_median_cache_placement(
    topology, cache_budget, n_cache_nodes, hit_ratio, weight="delay", **kwargs
):
    """Deploy caching nodes in locations that minimize overall latency assuming
    a partitioned strategy (a la Google Global Cache). According to this, in
    the network, a set of caching nodes are deployed and each receiver is
    mapped to one and only one caching node. Requests from this receiver are
    always sent to the designated caching node. In case of cache miss requests
    are forwarded to the original source.

    This placement problem can be mapped to the p-median location-allocation
    problem. This function solves this problem using the vertex substitution
    heuristic, which practically works like the k-medoid PAM algorithms, which
    is also similar to the k-means clustering algorithm. The result is not
    guaranteed to be globally optimal, only locally optimal.

    Notes
    -----
    This placement assumes that all receivers have degree = 1 and are connected
    to an ICR candidate nodes. Also, it assumes that contents are uniformly
    assigned to sources.

    Parameters
    ----------
    topology : Topology
        The topology object
    cache_budget : int
        The cumulative cache budget
    n_nodes : int
        The number of caching nodes to deploy
    hit_ratio : float
        The expected cache hit ratio of a single cache
    weight : str
        The weight attribute
    """
    n_cache_nodes = int(n_cache_nodes)
    icr_candidates = topology.graph["icr_candidates"]
    if len(icr_candidates) < n_cache_nodes:
        raise ValueError(
            "The number of ICR candidates (%d) is lower than "
            "the target number of caches (%d)" % (len(icr_candidates), n_cache_nodes)
        )
    elif len(icr_candidates) == n_cache_nodes:
        caches = list(icr_candidates)
        cache_assignment = {
            v: list(topology.adj[v].keys())[0] for v in topology.receivers()
        }
    else:
        # Need to optimally allocate caching nodes
        distances = dict(nx.all_pairs_dijkstra_path_length(topology, weight=weight))
        sources = topology.sources()
        d = {u: {} for u in icr_candidates}
        for u in icr_candidates:
            source_dist = sum(distances[u][source] for source in sources) / len(sources)
            for v in icr_candidates:
                if v in d[u]:
                    d[v][u] = d[u][v]
                else:
                    d[v][u] = distances[v][u] + (hit_ratio * source_dist)
        allocation, caches, _ = compute_p_median(distances, n_cache_nodes)
        cache_assignment = {
            v: allocation[list(topology.adj[v].keys())[0]] for v in topology.receivers()
        }

    cache_size = iround(cache_budget / n_cache_nodes)
    if cache_size == 0:
        raise ValueError(
            "Cache budget is %d but it's too small to deploy it on %d nodes. "
            "Each node will have a zero-sized cache. "
            "Set a larger cache budget and try again" % (cache_budget, n_cache_nodes)
        )
    for v in caches:
        topology.node[v]["stack"][1]["cache_size"] = cache_size
    topology.graph["cache_assignment"] = cache_assignment


@register_cache_placement("OPTIMAL_HASHROUTING")
def optimal_hashrouting_cache_placement(
    topology, cache_budget, n_cache_nodes, hit_ratio, weight="delay", **kwargs
):
    """Deploy caching nodes for hashrouting in optimized location

    Parameters
    ----------
    topology : Topology
        The topology object
    cache_budget : int
        The cumulative cache budget
    n_nodes : int
        The number of caching nodes to deploy
    hit_ratio : float
        The expected global cache hit ratio
    weight : str, optional
        The weight attribute. Default is 'delay'

    References
    ----------
    .. [1] L. Saino, I. Psaras and G. Pavlou, Framework and Algorithms for
           Operator-managed Content Caching, in IEEE Transactions on
           Network and Service Management (TNSM), Volume 17, Issue 1, March 2020
           https://doi.org/10.1109/TNSM.2019.2956525
    .. [2] L. Saino, On the Design of Efficient Caching Systems, Ph.D. thesis
           University College London, Dec. 2015. Available:
           http://discovery.ucl.ac.uk/1473436/
    """
    n_cache_nodes = int(n_cache_nodes)
    icr_candidates = topology.graph["icr_candidates"]
    if len(icr_candidates) < n_cache_nodes:
        raise ValueError(
            "The number of ICR candidates (%d) is lower than "
            "the target number of caches (%d)" % (len(icr_candidates), n_cache_nodes)
        )
    elif len(icr_candidates) == n_cache_nodes:
        caches = list(icr_candidates)
    else:
        # Need to optimally allocate caching nodes
        distances = dict(nx.all_pairs_dijkstra_path_length(topology, weight=weight))
        d = {}
        for v in icr_candidates:
            d[v] = 0
            for r in topology.receivers():
                d[v] += distances[r][v]
            for s in topology.sources():
                d[v] += distances[v][s] * hit_ratio

        # Sort caches in increasing order of distances and assign cache sizes
        caches = sorted(icr_candidates, key=lambda k: d[k])
    cache_size = iround(cache_budget / n_cache_nodes)
    if cache_size == 0:
        raise ValueError(
            "Cache budget is %d but it's too small to deploy it on %d nodes. "
            "Each node will have a zero-sized cache. "
            "Set a larger cache budget and try again" % (cache_budget, n_cache_nodes)
        )
    for v in caches[:n_cache_nodes]:
        topology.node[v]["stack"][1]["cache_size"] = cache_size


@register_cache_placement("CLUSTERED_HASHROUTING")
def clustered_hashrouting_cache_placement(
    topology, cache_budget, n_clusters, policy, distance="delay", **kwargs
):
    """Deploy caching nodes for hashrouting in with clusters

    Parameters
    ----------
    topology : Topology
        The topology object
    cache_budget : int
        The cumulative cache budget
    n_clusters : int
        The number of clusters
    policy : str (node_const | cluster_const)
        The expected global cache hit ratio
    distance : str
        The attribute used to quantify distance between pairs of nodes.
        Default is 'delay'

    References
    ----------
    .. [1] L. Saino, I. Psaras and G. Pavlou, Framework and Algorithms for
           Operator-managed Content Caching, in IEEE Transactions on
           Network and Service Management (TNSM), Volume 17, Issue 1, March 2020
           https://doi.org/10.1109/TNSM.2019.2956525
    .. [2] L. Saino, On the Design of Efficient Caching Systems, Ph.D. thesis
           University College London, Dec. 2015. Available:
           http://discovery.ucl.ac.uk/1473436/
    """
    icr_candidates = topology.graph["icr_candidates"]
    if n_clusters <= 0 or n_clusters > len(icr_candidates):
        raise ValueError(
            "The number of cluster must be positive and <= the "
            "number of ICR candidate nodes"
        )
    elif n_clusters == 1:
        clusters = [set(icr_candidates)]
    elif n_clusters == len(icr_candidates):
        clusters = [{v} for v in icr_candidates]
    else:
        clusters = compute_clusters(
            topology, n_clusters, distance=distance, nbunch=icr_candidates, n_iter=100
        )
    deploy_clusters(topology, clusters, assign_src_rcv=True)
    if policy == "node_const":
        # Each node is assigned the same amount of caching space
        cache_size = iround(cache_budget / len(icr_candidates))
        if cache_size == 0:
            return
        for v in icr_candidates:
            topology.node[v]["stack"][1]["cache_size"] = cache_size
    elif policy == "cluster_const":
        cluster_cache_size = iround(cache_budget / n_clusters)
        for cluster in topology.graph["clusters"]:
            cache_size = iround(cluster_cache_size / len(cluster))
            for v in cluster:
                if v not in icr_candidates:
                    continue
                topology.node[v]["stack"][1]["cache_size"] = cache_size
    else:
        raise ValueError("clustering policy %s not supported" % policy)


@register_cache_placement("GREEN")
def green_cache_placement(topology, cache_budget, **kwargs):
    """Places cache budget uniformly across cache nodes.

    Parameters
    ----------
    topology : Topology
        The topology object
    cache_budget : int
        The cumulative cache budget        
    """
    
    seed = kwargs.get("seed", 1.0)
    RGN = kwargs.get("RGN", 0.2)
    random.seed(seed)

    icr_candidates = topology.graph.get("icr_candidates", list(topology.nodes))
    if not icr_candidates:
        raise ValueError("No ICR candidates found in topology.")

    greenness = {}
    for node in icr_candidates:
        carbon_intensity = topology.nodes[node].get("carbon_intensity", 400)
        greenness[node] = carbon_intensity  # lower is better!

    # Select the greenest nodes (lowest carbon intensity)
    num_greens = max(1, int(len(icr_candidates) * RGN))  # at least 1 node
    sorted_nodes = sorted(greenness, key=greenness.get)  # ascending
    green_nodes = sorted_nodes[:num_greens]
    print(f"icr_candidates:{len(icr_candidates)}, num_green:{num_greens}")
   # Determine uniform cache size for selected green nodes
    cache_size = iround(cache_budget / num_greens)
    if cache_size == 0:
        raise ValueError(
            f"Cache budget ({cache_budget}) too small for {num_greens} green nodes. "
            f"Each would get zero cache. Increase budget or reduce RGN."
        )

    for v in green_nodes:
        topology.node[v]["stack"][1]["cache_size"] = cache_size


@register_cache_placement("HYBRID_GREEN_CENTRALITY")
def hybrid_cache_placement(topology, cache_budget, **kwargs):
    """
    Cache placement based on a hybrid score of greenness (low carbon intensity)
    and betweenness centrality. You control the balance with 'alpha'.

    Improvements:
    - Greenness normalized using min/max scaling.
    - Centrality normalized to [0,1].
    - Stochastic rounding for fairer allocations.
    - Uses topology.nodes instead of deprecated topology.node.
    """

    alpha = kwargs.get("alpha", 0.5)
    seed = kwargs.get("seed", 1)
    random.seed(seed)

    icr_candidates = topology.graph.get("icr_candidates", list(topology.nodes))
    if not icr_candidates:
        raise ValueError("No ICR candidates found in topology.")

    # 1. Greenness (lower carbon intensity is better → normalized score)
    carbon_intensities = {
        node: topology.nodes[node].get("carbon_intensity", 400)
        for node in icr_candidates
    }
    min_ci, max_ci = min(carbon_intensities.values()), max(carbon_intensities.values())
    greenness = {
        node: (max_ci - ci) / (max_ci - min_ci + 1e-9)
        for node, ci in carbon_intensities.items()
    }

    # 2. Betweenness centrality (normalized)
    betw = nx.betweenness_centrality(topology, normalized=True)
    centralities = {v: betw.get(v, 0.0) for v in icr_candidates}

    # 3. Combine into hybrid score
    scores = {
        node: alpha * greenness[node] + (1 - alpha) * centralities[node]
        for node in icr_candidates
    }

    # 4. Normalize scores to allocate proportionally
    total_score = sum(scores.values()) or 1e-9
    raw_alloc = {v: cache_budget * scores[v] / total_score for v in scores}

    # 5. Stochastic rounding + ensure ≥1
    rounded_alloc = {}
    for node, val in raw_alloc.items():
        base = int(val)
        frac = val - base
        rounded = base + (1 if random.random() < frac else 0)
        rounded_alloc[node] = max(1, rounded)

    # Adjust to exact budget if needed
    total_allocated = sum(rounded_alloc.values())
    while total_allocated > cache_budget:
        # Reduce from node with lowest score that has >1
        over_nodes = [n for n, a in rounded_alloc.items() if a > 1]
        if not over_nodes:
            break
        victim = min(over_nodes, key=lambda n: scores[n])
        rounded_alloc[victim] -= 1
        total_allocated -= 1
    while total_allocated < cache_budget:
        # Add to node with highest score
        winner = max(rounded_alloc, key=lambda n: scores[n])
        rounded_alloc[winner] += 1
        total_allocated += 1

    # 6. Apply cache sizes
    for node in icr_candidates:
        topology.nodes[node]["stack"][1]["cache_size"] = rounded_alloc[node]


@register_cache_placement("ALLOCATED")
def allocated_cache_placement(topology, cache_budget, allocations=None, **kwargs):
    """Assigns cache to ICR candidates based on user-provided allocations.

    - Supports allocations as a list (order = icr_candidates order).
    - Supports allocations as a dict {node: weight or count}.
    - If allocations sum to 1.0, they are treated as fractions of cache_budget.
    - Otherwise, they are treated as weights and normalized to cache_budget.
    - Ensures Icarus never sees cache_size=0 on active nodes.
    """
    icr_candidates = list(topology.graph["icr_candidates"])
    if allocations is None:
        raise ValueError("ALLOCATED placement requires 'allocations'")

    # --- Handle list case
    if isinstance(allocations, list):
        if len(allocations) != len(icr_candidates):
            raise ValueError(
                f"Allocations list length {len(allocations)} != icr_candidates length {len(icr_candidates)}"
            )
        allocations = {v: a for v, a in zip(icr_candidates, allocations)}

    # --- Active nodes only
    active_allocs = {v: a for v, a in allocations.items() if a > 0}
    if not active_allocs:
        return

    total_alloc = sum(active_allocs.values())

    # --- Normalize allocations
    if np.isclose(total_alloc, 1.0):
        # Treat as fractions
        norm_alloc = {v: allocations[v] * cache_budget for v in icr_candidates}
    else:
        # Treat as weights
        norm_alloc = {v: (allocations[v] / total_alloc) * cache_budget for v in icr_candidates}

    # --- Round, enforce ≥1 for active, adjust total
    rounded_alloc = {v: max(1, int(round(norm_alloc.get(v, 0)))) for v in active_allocs}
    total_allocated = sum(rounded_alloc.values())

    # Fix over-allocation (like in your betweenness code)
    while total_allocated > cache_budget:
        over_nodes = [v for v in rounded_alloc if rounded_alloc[v] > 1]
        if not over_nodes:
            break
        victim = min(over_nodes, key=lambda v: allocations[v])
        rounded_alloc[victim] -= 1
        total_allocated -= 1

    # --- Apply allocations to topology
    for v in icr_candidates:
        if v in rounded_alloc:
            topology.node[v]["stack"][1]["cache_size"] = rounded_alloc[v]
        else:
            if "cache_size" in topology.node[v]["stack"][1]:
                del topology.node[v]["stack"][1]["cache_size"]
