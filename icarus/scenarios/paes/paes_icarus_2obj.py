import logging
import os
import pickle
from pathlib import Path
import copy
import random

from icarus.scenarios.paes.paes import PAES
from icarus.runner import run
import networkx as nx

from icarus.util import iround
from multiprocessing import Pool


EXAMPLES_DIR = Path(__file__).parent

logger = logging.getLogger("babel")


# ================== top3 init ========================
def init_fn_top3(icr_candidates, params, allocs, cache_budget):
    net_params = params.get("network", {})
    ci = dict(net_params.get("node_carbon_intensity", {}))
    if not ci:
        raise ValueError("Carbon-aware init requires 'network.node_carbon_intensity'")

    topology = net_params.get("nx_graph")
    betw = nx.betweenness_centrality(topology)

    centralities = {n: betw.get(n, 0.0) for n in icr_candidates}

    scores = []
    for idx, node in enumerate(icr_candidates):
        carbon_score = -ci.get(node, 1.0)      # lower CI = better
        cent_score = centralities[node]
        score = 0.6 * carbon_score + 0.4 * cent_score
        scores.append((idx, score))

    k = min(3, len(scores))
    topk = sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    allocs = [0] * len(icr_candidates)
    base = cache_budget // k
    remainder = cache_budget % k

    for j, (idx, _) in enumerate(topk):
        allocs[idx] = base + (remainder if j == 0 else 0)

    tiers_template = params.get("cache_policy", {}).get("tiers", {})
    tiers_per_node = {n: copy.deepcopy(tiers_template) for n in icr_candidates}

    return {
        "allocations": allocs,
        "cache_budget": cache_budget,
        "actual_total": sum(allocs),
        "icr_candidates": icr_candidates,
        "tiers_per_node": tiers_per_node,
        "node_carbon_intensity": ci,
        "centralities": centralities,
    }

# ================== one shot init ========================
def init_fn_one_shot(icr_candidates, params, allocs, cache_budget):
    allocs = [0] * len(icr_candidates)
    winner_idx = random.randrange(len(icr_candidates))
    allocs[winner_idx] = cache_budget

    net_params = params.get("network", {})
    ci = dict(net_params.get("node_carbon_intensity", {}))
    if not ci:
        raise ValueError("Carbon-aware init requires 'network.node_carbon_intensity'")

    topology = net_params.get("nx_graph")
    betw = nx.betweenness_centrality(topology)
    centralities = {n: betw.get(n, 0.0) for n in icr_candidates}

    tiers_template = params.get("cache_policy", {}).get("tiers", {})
    tiers_per_node = {n: copy.deepcopy(tiers_template) for n in icr_candidates}

    return {
        "allocations": allocs,
        "cache_budget": cache_budget,
        "actual_total": sum(allocs),
        "icr_candidates": icr_candidates,
        "tiers_per_node": tiers_per_node,
        "node_carbon_intensity": ci,
        "centralities": centralities,
    }

# ================== uniform init ========================
def init_fn_uniform(icr_candidates, params, allocs, cache_budget):
    cache_size = iround(cache_budget / len(icr_candidates))
    
    raw_alloc = {
        v: cache_size
        for v in icr_candidates
    }
    
    # Initial rounding with minimum 0
    rounded_alloc = {v: max(0, int(round(a))) for v, a in raw_alloc.items()}
    
    allocs[:] = rounded_alloc.values()

    # 8) Tier setup (unchanged from your pattern)
    tiers_template = params.get("cache_policy", {}).get("tiers", {})
    tiers_per_node = {node: copy.deepcopy(tiers_template) for node in icr_candidates}
    
    ci = params.get("network").get("node_carbon_intensity")
    ci = dict(ci)
    if not ci:
        raise ValueError("Carbon-aware init requires 'network.node_carbon_intensity'")
    
    net_params = params.get("network", {})
    topology = net_params.get("nx_graph")
    betw = dict(nx.betweenness_centrality(topology))

    central = {v: betw[v] for v in icr_candidates}
    return {
        "allocations": allocs[:],
        "cache_budget": cache_budget,
        "actual_total": sum(rounded_alloc.values()),
        "icr_candidates": icr_candidates,
        "tiers_per_node": tiers_per_node,
        "node_carbon_intensity": ci,
        "centralities": central
    }

# ================== ci only init ========================
def init_fn_ci_only(icr_candidates, params, allocs, cache_budget):
    centralities = params.get("network").get("node_carbon_intensity")
    centralities = dict(centralities)
    if not centralities:
        raise ValueError("Carbon-aware init requires 'network.node_carbon_intensity'")

    total_centrality = sum(centralities.values())

    raw_alloc = {
        v: cache_budget * centralities[v] / total_centrality
        for v in centralities
    }
    
    # Initial rounding with minimum 0
    rounded_alloc = {v: max(0, int(round(a))) for v, a in raw_alloc.items()}
    total_allocated = sum(rounded_alloc.values())
    while total_allocated > cache_budget:
        # Find the node with the smallest allocation > 1 to reduce
        over_nodes = [v for v in rounded_alloc if rounded_alloc[v] > 1]
        if not over_nodes:
            break  # Can't reduce anymore without violating ≥1 constraint
        # Reduce the one with the smallest centrality
        victim = min(over_nodes, key=lambda v: centralities[v])
        rounded_alloc[victim] -= 1
        total_allocated -= 1
    
    allocs[:] = rounded_alloc.values()

    # 8) Tier setup (unchanged from your pattern)
    tiers_template = params.get("cache_policy", {}).get("tiers", {})
    tiers_per_node = {node: copy.deepcopy(tiers_template) for node in icr_candidates}
    net_params = params.get("network", {})
    topology = net_params.get("nx_graph")
    betw = dict(nx.betweenness_centrality(topology))

    central = {v: betw[v] for v in icr_candidates}
    return {
        "allocations": allocs[:],
        "cache_budget": cache_budget,
        "actual_total": sum(rounded_alloc.values()),
        "icr_candidates": icr_candidates,
        "tiers_per_node": tiers_per_node,
        "node_carbon_intensity": centralities,
        "centralities": central
    }

# ================== betweenness centrality only init ========================
def init_fn_bc_only(icr_candidates, params, allocs, cache_budget):
    net_params = params.get("network", {})
    topology = net_params.get("nx_graph")
    # betw = dict(nx.betweenness_centrality(topology))
    # pr_kwargs = {}
    # betw = dict(nx.pagerank(topology, **pr_kwargs))
    betw = dict(nx.betweenness_centrality(topology))

    centralities = {v: betw[v] for v in icr_candidates}
    # centralities = {v: betw[v] for v in icr_candidates if betw[v] > 0}
    if not centralities:
        raise ValueError("No centralities")

    total_centrality = sum(centralities.values())

    raw_alloc = {
        v: cache_budget * centralities[v] / total_centrality
        for v in centralities
    }
    
    # Initial rounding with minimum 0
    rounded_alloc = {v: max(0, int(round(a))) for v, a in raw_alloc.items()}
    total_allocated = sum(rounded_alloc.values())
    while total_allocated > cache_budget:
        # Find the node with the smallest allocation > 1 to reduce
        over_nodes = [v for v in rounded_alloc if rounded_alloc[v] > 1]
        if not over_nodes:
            break  # Can't reduce anymore without violating ≥1 constraint
        # Reduce the one with the smallest centrality
        victim = min(over_nodes, key=lambda v: centralities[v])
        rounded_alloc[victim] -= 1
        total_allocated -= 1
    
    allocs[:] = rounded_alloc.values()
    # print(allocs)

    # 8) Tier setup (unchanged from your pattern)
    tiers_template = params.get("cache_policy", {}).get("tiers", {})
    tiers_per_node = {node: copy.deepcopy(tiers_template) for node in icr_candidates}
    
    ci = params.get("network").get("node_carbon_intensity")
    ci = dict(ci)
    if not ci:
        raise ValueError("Carbon-aware init requires 'network.node_carbon_intensity'")

    return {
        "allocations": allocs[:],
        "cache_budget": cache_budget,
        "actual_total": sum(rounded_alloc.values()),
        "icr_candidates": icr_candidates,
        "tiers_per_node": tiers_per_node,
        "node_carbon_intensity": ci,
        "centralities": centralities
    }

# ================== degree centrality only init ========================
def init_fn_dc_only(icr_candidates, params, allocs, cache_budget):
    net_params = params.get("network", {})
    topology = net_params.get("nx_graph")
    # betw = dict(nx.betweenness_centrality(topology))
    # pr_kwargs = {}
    # betw = dict(nx.pagerank(topology, **pr_kwargs))
    betw = dict(nx.degree(topology))

    centralities = {v: betw[v] for v in icr_candidates}
    # centralities = {v: betw[v] for v in icr_candidates if betw[v] > 0}
    if not centralities:
        raise ValueError("No centralities")

    total_centrality = sum(centralities.values())

    raw_alloc = {
        v: cache_budget * centralities[v] / total_centrality
        for v in centralities
    }
    
    # Initial rounding with minimum 0
    rounded_alloc = {v: max(0, int(round(a))) for v, a in raw_alloc.items()}
    total_allocated = sum(rounded_alloc.values())
    while total_allocated > cache_budget:
        # Find the node with the smallest allocation > 1 to reduce
        over_nodes = [v for v in rounded_alloc if rounded_alloc[v] > 1]
        if not over_nodes:
            break  # Can't reduce anymore without violating ≥1 constraint
        # Reduce the one with the smallest centrality
        victim = min(over_nodes, key=lambda v: centralities[v])
        rounded_alloc[victim] -= 1
        total_allocated -= 1
    
    allocs[:] = rounded_alloc.values()
    # print(allocs)

    # 8) Tier setup (unchanged from your pattern)
    tiers_template = params.get("cache_policy", {}).get("tiers", {})
    tiers_per_node = {node: copy.deepcopy(tiers_template) for node in icr_candidates}
    
    ci = params.get("network").get("node_carbon_intensity")
    ci = dict(ci)
    if not ci:
        raise ValueError("Carbon-aware init requires 'network.node_carbon_intensity'")

    return {
        "allocations": allocs[:],
        "cache_budget": cache_budget,
        "actual_total": sum(rounded_alloc.values()),
        "icr_candidates": icr_candidates,
        "tiers_per_node": tiers_per_node,
        "node_carbon_intensity": ci,
        "centralities": centralities
    }

# ================== pagerank centrality only init ========================
def init_fn_pr_only(icr_candidates, params, allocs, cache_budget):
    net_params = params.get("network", {})
    topology = net_params.get("nx_graph")
    pr_kwargs = {}
    betw = dict(nx.pagerank(topology, **pr_kwargs))

    centralities = {v: betw[v] for v in icr_candidates}
    # centralities = {v: betw[v] for v in icr_candidates if betw[v] > 0}
    if not centralities:
        raise ValueError("No centralities")

    total_centrality = sum(centralities.values())

    raw_alloc = {
        v: cache_budget * centralities[v] / total_centrality
        for v in centralities
    }
    
    # Initial rounding with minimum 0
    rounded_alloc = {v: max(0, int(round(a))) for v, a in raw_alloc.items()}
    total_allocated = sum(rounded_alloc.values())
    while total_allocated > cache_budget:
        # Find the node with the smallest allocation > 1 to reduce
        over_nodes = [v for v in rounded_alloc if rounded_alloc[v] > 1]
        if not over_nodes:
            break  # Can't reduce anymore without violating ≥1 constraint
        # Reduce the one with the smallest centrality
        victim = min(over_nodes, key=lambda v: centralities[v])
        rounded_alloc[victim] -= 1
        total_allocated -= 1
    
    allocs[:] = rounded_alloc.values()
    # print(allocs)

    # 8) Tier setup (unchanged from your pattern)
    tiers_template = params.get("cache_policy", {}).get("tiers", {})
    tiers_per_node = {node: copy.deepcopy(tiers_template) for node in icr_candidates}
    
    ci = params.get("network").get("node_carbon_intensity")
    ci = dict(ci)
    if not ci:
        raise ValueError("Carbon-aware init requires 'network.node_carbon_intensity'")

    return {
        "allocations": allocs[:],
        "cache_budget": cache_budget,
        "actual_total": sum(rounded_alloc.values()),
        "icr_candidates": icr_candidates,
        "tiers_per_node": tiers_per_node,
        "node_carbon_intensity": ci,
        "centralities": centralities
    }

# ================== alpha * ci + (1 - alpha) * centrality ========================
def init_fn_alpha(icr_candidates, params, allocs, cache_budget):
    net_params = params.get("network", {})
    topology = net_params.get("nx_graph")
    ci = net_params.get("node_carbon_intensity")
    if topology is None or ci is None:
        raise ValueError("Init requires 'network.nx_graph' and 'network.node_carbon_intensity'")

    # weight between CI and centrality
    alpha = params.get("green", {}).get("alpha", 0.5)

    # 1) topology centrality (degree here; swap to betweenness/pagerank if you like)
    deg = dict(nx.betweenness_centrality(topology))

    # restrict to candidates with positive degree
    centralities = {v: float(deg[v]) for v in icr_candidates}
    if not centralities:
        raise ValueError("No centralities")

    # 2) normalize centrality to [0,1]
    cent_vals = list(centralities.values())
    cent_min = min(cent_vals)
    cent_max = max(cent_vals)
    cent_range = cent_max - cent_min or 1.0
    cent_norm = {v: (centralities[v] - cent_min) / cent_range for v in centralities}

    # 3) normalize CI so that cleaner (lower CI) = higher score in [0,1]
    ci_sub = {v: float(ci[v]) for v in centralities if v in ci}
    if not ci_sub:
        raise ValueError("No CI entries for the candidate nodes")
    ci_vals = list(ci_sub.values())
    ci_min = min(ci_vals)
    ci_max = max(ci_vals)
    ci_range = ci_max - ci_min or 1.0
    # inverted & normalized: low CI → 1, high CI → 0
    ci_norm = {v: (ci_max - ci_sub[v]) / ci_range for v in ci_sub}

    # ensure all centrality nodes have a CI score (fallback 0 if missing)
    for v in centralities:
        ci_norm.setdefault(v, 0.0)

    # 4) combined score: alpha * ci_norm + (1 - alpha) * cent_norm
    scores = {
        v: alpha * ci_norm[v] + (1.0 - alpha) * cent_norm[v]
        for v in centralities
    }

    # drop any nodes with non-positive score
    scores = {v: s for v, s in scores.items() if s > 0.0}
    if not scores:
        return {
            "allocations": {},
            "cache_budget": cache_budget,
            "actual_total": 0,
            "icr_candidates": icr_candidates,
            "tiers_per_node": {},
            "node_carbon_intensity": ci,
            "centralities": centralities
        }

    total_score = sum(scores.values())

    # 5) proportional allocation on combined score
    raw_alloc = {
        v: cache_budget * scores[v] / total_score
        for v in scores
    }

    rounded_alloc = {v: max(0, int(round(a))) for v, a in raw_alloc.items()}
    total_allocated = sum(rounded_alloc.values())

    # if we overshoot the budget, decrement lowest-score nodes first
    while total_allocated > cache_budget:
        over_nodes = [v for v in rounded_alloc if rounded_alloc[v] > 1]
        if not over_nodes:
            break
        victim = min(over_nodes, key=lambda v: scores[v])
        rounded_alloc[victim] -= 1
        total_allocated -= 1

    # if we undershoot (because of rounding), give leftover to highest-score nodes
    while total_allocated < cache_budget:
        beneficiary = max(rounded_alloc.keys(), key=lambda v: scores[v])
        rounded_alloc[beneficiary] += 1
        total_allocated += 1

    # map allocs[] (which is ordered like icr_candidates) from the dict
    alloc_map = {v: 0 for v in icr_candidates}
    alloc_map.update(rounded_alloc)
    allocs[:] = [alloc_map[v] for v in icr_candidates]

    tiers_template = params.get("cache_policy", {}).get("tiers", {})
    tiers_per_node = {node: copy.deepcopy(tiers_template) for node in icr_candidates}

    return {
        "allocations": allocs[:],
        "cache_budget": cache_budget,
        "actual_total": total_allocated,
        "icr_candidates": icr_candidates,
        "tiers_per_node": tiers_per_node,
        "node_carbon_intensity": ci,
        "centralities": centralities
    }

# ================== hub fraction ========================
def init_fn_hub(icr_candidates, params, allocs, cache_budget):
    net_params = params.get("network", {})
    topology = net_params.get("nx_graph")
    ci = net_params.get("node_carbon_intensity")
    if topology is None or ci is None:
        raise ValueError("Init requires 'network.nx_graph' and 'network.node_carbon_intensity'")
    
    # Tunable parameters
    hub_fraction = params.get("network", {}).get("hub_fraction")  # % of nodes as hubs
    hub_share = params.get("network", {}).get("hub_budget_share")  # Budget % to hubs

    deg = dict(nx.betweenness_centrality(topology))  # degree centrality related to importance [web:27]
    centralities = {v: float(deg.get(v, 0.0)) for v in icr_candidates}
    # filter positive centrality
    centralities = {v: c for v, c in centralities.items()}
    
    if not centralities:
        raise ValueError("No centralities")
    
    nodes = list(centralities.keys())
    n = len(nodes)

    k = max(1, int(round(hub_fraction * n)))
    # sort by centrality descending
    nodes_sorted = sorted(nodes, key=lambda v: centralities[v], reverse=True)
    hub_nodes = set(nodes_sorted[:k])
    non_hub_nodes = set(nodes) - hub_nodes

    hub_budget = cache_budget * hub_share
    non_hub_budget = cache_budget - hub_budget

    if hub_nodes:
        hub_cents = {v: centralities[v] for v in hub_nodes}
        total_hub_cent = sum(hub_cents.values())
        # if degenerate, fall back to uniform among hubs
        if total_hub_cent <= 0:
            hub_raw = {v: hub_budget / len(hub_nodes) for v in hub_nodes}
        else:
            hub_raw = {v: hub_budget * hub_cents[v] / total_hub_cent for v in hub_nodes}
    else:
        hub_raw = {}

    # Phase 2: Allocate remainder to non-hubs using CI only
    if non_hub_nodes:
        ci_non = {v: float(ci[v]) for v in non_hub_nodes if v in ci}
        if not ci_non:
            # if no CI info, uniform among non-hubs
            non_hub_raw = {v: non_hub_budget / len(non_hub_nodes) for v in non_hub_nodes}
        else:
            ci_vals = list(ci_non.values())
            ci_min = min(ci_vals)
            ci_max = max(ci_vals)
            ci_range = ci_max - ci_min or 1.0
            # inverted & normalized: low CI → high score
            ci_score = {v: (ci_max - ci_non[v]) / ci_range for v in ci_non}
            # ensure all non-hubs have some score (0 if missing)
            for v in non_hub_nodes:
                ci_score.setdefault(v, 0.0)
            total_ci_score = sum(ci_score.values())
            if total_ci_score <= 0:
                non_hub_raw = {v: non_hub_budget / len(non_hub_nodes) for v in non_hub_nodes}
            else:
                non_hub_raw = {
                    v: non_hub_budget * ci_score[v] / total_ci_score
                    for v in non_hub_nodes
                }
    else:
        non_hub_raw = {}
    
    # 6) merge raw allocations and round, enforcing ≥1 for any node that gets non-zero
    raw_alloc = {}
    raw_alloc.update(hub_raw)
    raw_alloc.update(non_hub_raw)

    # only keep nodes with positive raw allocation
    raw_alloc = {v: a for v, a in raw_alloc.items() if a > 0.0}

    if not raw_alloc:
        raise ValueError("No raw_alloc")

    rounded_alloc = {v: max(0, int(round(a))) for v, a in raw_alloc.items()}
    total_allocated = sum(rounded_alloc.values())

    # 7) adjust down if over budget: remove from lowest-priority nodes
    # priority: non-hubs first (penalize greener fringe before hubs), then lowest centrality
    def priority_key(v):
        # higher = better; we want to remove from lowest
        is_hub = 1 if v in hub_nodes else 0
        return (is_hub, centralities.get(v, 0.0))
   
    while total_allocated > cache_budget:
        over_nodes = [v for v in rounded_alloc if rounded_alloc[v] > 1]
        if not over_nodes:
            break
        victim = min(over_nodes, key=priority_key)  # lowest priority
        rounded_alloc[victim] -= 1
        total_allocated -= 1

    # 8) if under budget (rare), add to highest priority nodes (hubs with high centrality)
    while total_allocated < cache_budget:
        beneficiary = max(rounded_alloc.keys(), key=priority_key)
        rounded_alloc[beneficiary] += 1
        total_allocated += 1

    # 9) map back to allocs[] order (icr_candidates)
    alloc_map = {v: 0 for v in icr_candidates}
    alloc_map.update(rounded_alloc)
    allocs[:] = [alloc_map[v] for v in icr_candidates]

    tiers_template = params.get("cache_policy", {}).get("tiers", {})
    tiers_per_node = {node: copy.deepcopy(tiers_template) for node in icr_candidates}
    
    return {
        "allocations": allocs[:],
        "cache_budget": cache_budget,
        "actual_total": total_allocated,
        "icr_candidates": icr_candidates,
        "tiers_per_node": tiers_per_node,
        "node_carbon_intensity": ci,
        "centralities": centralities
    }

# --- Mutation function ---
# def mutate_fn(sol, rng):
#     new_sol = copy.deepcopy(sol)
#     alloc = new_sol["allocations"][:]
#     n = len(alloc)
#     nodes = list(range(n))
#     budget = new_sol["cache_budget"]
#     icr = new_sol["icr_candidates"]
#     icr = list(icr)

#     # ---- parameters ----
#     modes = ["carbon"]*3 + ["centrality"]*3 + ["random"]*1  
#     mode = rng.choice(modes)
#     print(f"{mode} MUTATE")

#     # mode-dependent mutation strength
#     if mode == "random":
#         move_frac = rng.uniform(0.15, 0.35)   # strong exploration
#     elif mode == "carbon":
#         move_frac = rng.uniform(0.08, 0.2)
#     else:  # hit
#         move_frac = rng.uniform(0.05, 0.15)

#     amount = max(0, int(budget * move_frac))

#     # ---- helpers ----
#     non_zero = [i for i in nodes if alloc[i] > 0]
#     if len(non_zero) < 2:
#         return new_sol

#     # ==========================================================
#     # MODE 1 — CARBON-AWARE (push cache toward clean nodes)
#     # ==========================================================
#     if mode == "carbon":
#         ci = new_sol["node_carbon_intensity"]
#         ranked = sorted(non_zero, key=lambda i: ci.get(icr[i]))
#         donors = ranked[-max(2, len(ranked)//4):]     # dirty nodes
#         receivers = ranked[:max(2, len(ranked)//4)]   # clean nodes

#     # ==========================================================
#     # MODE 2 — HIT-ORIENTED (reinforce strong caches)
#     # ==========================================================
#     elif mode == "centrality":
#         centralities = new_sol["centralities"]
#         ranked = sorted(non_zero, key=lambda i: centralities.get(icr[i]))
#         donors = ranked[:max(2, len(ranked)//4)]      # low-central nodes
#         receivers = ranked[-max(2, len(ranked)//4):]  # high-central nodes

#     # ==========================================================
#     # MODE 3 — RANDOM / SHAKE (escape local basin)
#     # ==========================================================
#     else:
#         # occasionally kill a node entirely
#         if rng.random() < 0.3:
#             victim = rng.choice(non_zero)
#             amount += alloc[victim]
#             alloc[victim] = 0
#             non_zero.remove(victim)

#         donors = rng.sample(non_zero, min(5, len(non_zero)))
#         receivers = rng.sample(nodes, rng.randint(4, n))

#     # ---- MOVE CACHE MASS ----
#     # steal
#     for d in donors:
#         if amount <= 0:
#             break
#         steal = min(alloc[d], max(1, amount // len(donors)))
#         alloc[d] -= steal
#         amount -= steal

#     # give
#     for r in receivers:
#         if amount <= 0:
#             break
#         give = max(1, amount // len(receivers))
#         alloc[r] += give
#         amount -= give

#     # ---- SNAP BUDGET EXACTLY ----
#     total = sum(alloc)
#     while total > budget:
#         i = rng.choice([i for i in nodes if alloc[i] > 0])
#         alloc[i] -= 1
#         total -= 1
#     while total < budget:
#         alloc[rng.choice(nodes)] += 1
#         total += 1

#     new_sol["allocations"] = alloc
#     return new_sol

def mutate_ci_strata_fixed(sol, rng):
    new = copy.deepcopy(sol)
    n = len(new["allocations"])
    nodes = list(range(n))
    budget = int(new["cache_budget"])
    icr = list(new["icr_candidates"])
    ci = new["node_carbon_intensity"]

    sorted_nodes = sorted(nodes, key=lambda i: ci[icr[i]])
    k = max(1, n // 3)
    low, mid, high = sorted_nodes[:k], sorted_nodes[k:2*k], sorted_nodes[2*k:]
    
    strata = {"low": low, "mid": mid, "high": high, "low_mid": low+mid, "all": nodes}
    target = strata[rng.choice(list(strata))]
    
    # 70% sparse (1-3 nodes), 30% dense
    if rng.random() < 0.7 and len(target) > 3:
        active = rng.sample(target, rng.randint(1, 3))
    else:
        active = target
    
    alloc = [0] * n
    for _ in range(budget):
        alloc[rng.choice(active)] += 1
    new["allocations"] = alloc
    return new

def mutate_centrality_shells_fixed(sol, rng):
    new = copy.deepcopy(sol)
    alloc = new["allocations"][:]
    n = len(alloc)
    nodes = list(range(n))
    budget = int(new["cache_budget"])
    icr = list(new["icr_candidates"])
    centralities = new["centralities"]
    
    ranked = sorted(nodes, key=lambda i: centralities[icr[i]], reverse=True)
    k = max(1, n // 3)
    core, ring, fringe = ranked[:k], ranked[k:2*k], ranked[2*k:]
    
    shells = {"core": core, "ring": ring, "fringe": fringe}
    src_shell = rng.choice(list(shells))
    dst_shell = rng.choice([s for s in shells if s != src_shell] or list(shells))
    
    src, dst = shells[src_shell], shells[dst_shell]
    move_frac = rng.uniform(0.4, 0.7)  # MORE AGGRESSIVE
    amount = max(1, int(budget * move_frac))
    
    src_nonzero = [i for i in src if alloc[i] > 0]
    if not src_nonzero: return new
    
    for d in src_nonzero:
        if amount <= 0: break
        steal = min(alloc[d], max(1, amount // max(1, len(src_nonzero))))
        alloc[d] -= steal
        amount -= steal
    
    while amount > 0 and dst:
        alloc[rng.choice(dst)] += 1
        amount -= 1
    
    new["allocations"] = alloc
    return new

def mutate_fn(sol, rng):
    new_sol = copy.deepcopy(sol)
    alloc = new_sol["allocations"][:]
    n = len(alloc)
    nodes = list(range(n))
    budget = int(new_sol["cache_budget"])
    icr = list(new_sol["icr_candidates"])
    ci = new_sol["node_carbon_intensity"]
    centralities = new_sol["centralities"]
    
    # ===== NUCLEAR EXPLORATION (50% total) =====
    r = rng.random()
    
    # # 🔥 20% ONE-HOT NUCLEAR (up from 15%)
    # if r < 0.20:
    #     winner = rng.choice(nodes)
    #     alloc = [0] * n
    #     alloc[winner] = budget
    #     print(f"💥 ONEHOT idx={winner}")
    #     new_sol["allocations"] = alloc
    #     return new_sol
    
    # # 🏆 15% TOP-3 DOMINATION (up from 10%)
    # elif r < 0.35:
    #     scores = []
    #     for i in nodes:
    #         c = -ci.get(icr[i], 1.0)
    #         cent = centralities.get(icr[i], 0)
    #         scores.append((i, 0.6*c + 0.4*cent))  # more CI bias
        
    #     top3 = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
    #     alloc = [0] * n
    #     base = budget // 3
    #     alloc[top3[0][0]] = base + budget % 3
    #     alloc[top3[1][0]] = base
    #     alloc[top3[2][0]] = base
    #     print("💣 TOP3 DOMINATION")
    #     new_sol["allocations"] = alloc
    #     return new_sol
    
    # 🌪️ 15% NEW: ZERO-5 ZONES (kill 14/19 nodes!)
    # elif r < 0.50:
    #     # Keep ONLY 1-5 nodes alive, random survivors
    #     survivors = rng.sample(nodes, rng.randint(1, 5))
    #     alloc = [0] * n
    #     for _ in range(budget):
    #         i = rng.choice(survivors)
    #         alloc[i] += 1
    #     print(f"☢️ ZERO-5: {len(survivors)} survivors")
    #     new_sol["allocations"] = alloc
    #     return new_sol
    
    # ===== CHAOS MUTATIONS (50% total) =====
    
    # NEW #1: INVERT ORDER (reverse allocation ranking)
    if rng.random() < 0.15:
        ranked = sorted(range(n), key=lambda i: alloc[i], reverse=True)
        inverted = ranked[::-1]  # worst become best
        alloc = [0] * n
        for rank, i in enumerate(inverted):
            share = max(1, budget // max(1, n-rank))
            alloc[i] = min(share, budget - sum(alloc))
            if sum(alloc) >= budget: break
        print("🔄 INVERT ORDER")
    
    # NEW #2: OPPOSITE EXTREMES (dirty+fringe vs clean+core)
    elif rng.random() < 0.15:
        ci_clean = sorted(nodes, key=lambda i: ci[icr[i]])[:3]
        ci_dirty = sorted(nodes, key=lambda i: ci[icr[i]], reverse=True)[:3]
        cent_core = sorted(nodes, key=lambda i: centralities[icr[i]], reverse=True)[:3]
        cent_fringe = sorted(nodes, key=lambda i: centralities[icr[i]])[:3]
        
        # 50/50: clean+fringe OR dirty+core (anti-intuitive!)
        if rng.random() < 0.5:
            targets = ci_clean + cent_fringe
            print("🧪 CLEAN+FRINGE")
        else:
            targets = ci_dirty + cent_core  
            print("🧪 DIRTY+CORE")
        
        alloc = [0] * n
        for _ in range(budget):
            i = rng.choice(list(set(targets)))
            alloc[i] += 1
        new_sol["allocations"] = alloc
        return new_sol
    
    # FIXED: ci_strata (no more undefined alpha)
    elif rng.random() < 0.10:
        return mutate_ci_strata_fixed(sol, rng)
    
    # FIXED: centrality_shells  
    elif rng.random() < 0.10:
        return mutate_centrality_shells_fixed(sol, rng)
    
    # MASSIVE SHAKE (40% budget moves!)
    else:
        move_frac = rng.uniform(0.35, 0.60)  # WAY more aggressive
        amount = max(1, int(budget * move_frac))
        
        # KILL 30% of active nodes completely
        non_zero = [i for i in nodes if alloc[i] > 0]
        victims = rng.sample(non_zero, max(0, int(len(non_zero) * 0.3)))
        for v in victims:
            amount += alloc[v]
            alloc[v] = 0
        
        # Donors: worst by combined score
        scores = {i: -ci.get(icr[i],1.0) + 0.3*centralities.get(icr[i],0) 
                 for i in non_zero if i not in victims}
        donors = sorted(scores, key=scores.get)[:max(6, n//3)]
        
        # Receivers: best by combined score + some chaos
        receivers = sorted(scores, key=scores.get, reverse=True)[:8]
        receivers += rng.sample(nodes, 4)  # chaos injection
        
        # Steal aggressively
        for d in donors:
            if amount <= 0: break
            steal = min(alloc[d], max(1, amount // max(1, len(donors))))
            alloc[d] -= steal
            amount -= steal
        
        # Distribute wildly
        for r in receivers:
            if amount <= 0: break
            give = max(1, amount // max(1, len(receivers)))
            alloc[r] += give
            amount -= give

    # SNAP BUDGET (allow zeros!)
    total = sum(alloc)
    while total > budget:
        candidates = [i for i in nodes if alloc[i] > 0]
        if candidates:
            i = rng.choice(candidates)
            alloc[i] -= 1
            total -= 1
    while total < budget:
        i = rng.choice(nodes)
        alloc[i] += 1
        total += 1
    
    new_sol["allocations"] = alloc
    return new_sol

def build_experiment(icr_candidates, params, metrics, allocations, tiers_per_node, cache_placement="ALLOCATED"):
    exp = copy.deepcopy(params)
    exp["data_collectors"] = copy.deepcopy(metrics)

    if allocations is not None:
        # PAES mode: override to ALLOCATED
        exp["cache_placement"]["name"] = "ALLOCATED"
        exp["cache_placement"]["allocations"] = {
            node: int(a) for node, a in zip(icr_candidates, allocations)
        }
    else:
        # Baseline mode: keep original params unchanged
        pass
    # exp["workload"]["n_warmup"] /= 2
    # exp["workload"]["n_measured"] /= 2
    # exp["workload"]["n_measured"] = int(exp["workload"]["n_measured"])
     # tiers logic unchanged
    if tiers_per_node is not None:
        exp["cache_policy"]["tiers_per_node"] = tiers_per_node
    
    return exp

def _call_init_worker(args):
    init_fn, icr_candidates, params, cache_budget = args
    allocs = [0] * len(icr_candidates)
    return init_fn(icr_candidates, params, allocs, cache_budget)

def batch_init_wrapper(pool, icr_candidates, params, cache_budget):
    def batch_init(init_fns):
        tasks = [(fn, icr_candidates, params, cache_budget) for fn in init_fns]
        async_results = [
            pool.apply_async(_call_init_worker, (t,))
            for t in tasks
        ]
        return [r.get() for r in async_results]
    return batch_init

def eval_fn(sol, **kwargs):
    params  = kwargs["params"]
    period = params["network"]["period"]
    prev_state_path = params["network"]["prev_state_path"]
    if prev_state_path!= None:
        prev_state_path = repr(str(prev_state_path))
    metrics = kwargs["metrics"]
    settings = kwargs["settings"]
    cache_fraction = sol["cache_budget"]   # or get from params/kwargs
    tag = f"cache{int(cache_fraction)}_pid{os.getpid()}_per{period}"
    base = EXAMPLES_DIR / tag
    results_file = base.with_suffix(".paes_results.pickle")
    config_file = base.with_suffix(".paes_config.py")
    exp_pkl = base.with_suffix(".paes.pkl")
    
    exp = build_experiment(
        icr_candidates=sol["icr_candidates"],
        params=params,
        metrics=metrics,
        allocations=sol["allocations"],
        tiers_per_node=sol["tiers_per_node"],
        cache_placement="ALLOCATED"
    )
    if hasattr(exp, "to_dict"):
        exp = exp.to_dict() 
    
    total = sum(sol["allocations"])
    logger.info(f"Evaluating allocations:{[a for a in sol['allocations']]}, sum : {total}")
    print(f"Evaluating allocations:{[a for a in sol['allocations']]}, sum : {total}")

    with open(exp_pkl, "wb") as f:
        pickle.dump([exp], f)

    with open(config_file, "w") as f:
        f.write("from collections import deque\n")
        f.write("import pickle\n")
        f.write(f"LOG_LEVEL = {repr(str(settings.LOG_LEVEL))}\n")
        f.write(f"CACHING_GRANULARITY = {repr(str(settings.CACHING_GRANULARITY))}\n")
        f.write(f"RESULTS_FORMAT = {repr(str(settings.RESULTS_FORMAT))}\n")
        f.write("PARALLEL_EXECUTION = False\n")
        f.write(f"N_REPLICATIONS = {settings.N_REPLICATIONS}\n")
        f.write(f"N_PERIODS = 1\n")
        f.write(f"GREEN_PERIOD = {period}\n")
        f.write(f"PREV_STATE_PATH = {prev_state_path}\n")
        f.write("EXPERIMENT_QUEUE = deque()\n")
        f.write(f"EXPERIMENT_QUEUE.extend(pickle.load(open({repr(str(exp_pkl))}, 'rb')))\n")

    try:
        run(str(config_file), str(results_file), {})
    except Exception as e:
        print("Icarus failed:", e)
        return (None, None, None)

    if not results_file.exists():
        return (None, None, None)

    try:
        with open(results_file, "rb") as f:
            data = pickle.load(f)
        
        _, metrics = data._results[0]
        hit = metrics.get("CACHE_HIT_RATIO").get("MEAN")
        cost = metrics.get("COST").get("MEAN")
        carbon = metrics.get("CARBONFOOTPRINT").get("TOTAL")
        logger.info(f"carbon : {carbon}, hit : {hit}, cost : {cost}, ")
        print(f"carbon : {carbon}, hit : {hit}, cost : {cost} ")

        # GREEN PENALTY: if too much cache on dirty nodes
        # try:
        #     ci = sol.get("node_carbon_intensity")
        #     nodes = sol.get("icr_candidates", [])
        #     allocs = sol.get("allocations", [])
        #     if ci and nodes and allocs:
        #         ci_vals = [ci.get(n, 1.0) for n in nodes]
        #         med_ci = statistics.median(ci_vals)
        #         total_alloc = sum(allocs) or 1

        #         dirty_indices = [i for i, n in enumerate(nodes) if ci.get(n, 1.0) > med_ci]
        #         dirty_alloc = sum(allocs[i] for i in dirty_indices)

        #         dirty_ratio = dirty_alloc / total_alloc
        #         if dirty_ratio > 0.5:
        #             carbon *= 10
        #             logger.warning(f"Hard elimination penalty applied (dirty_ratio={dirty_ratio:.3f})")
        # except Exception as e:
        #     logger.warning(f"Carbon penalty skipped due to error: {e}")
        return (carbon, hit, cost)
    except Exception as e:
        print("Error reading results:", e)
        return (None, None, None)

def eval_wrapper(args):
    sol, params, metrics, settings = args
    return eval_fn(sol, params=params, metrics=metrics, settings=settings)

def batch_eval_wrapper(pool, params, metrics, settings):
    """Top-level picklable batch evaluator. Takes a POOL instance."""
    def batch_eval(sols):
        futures = [pool.apply_async(eval_wrapper, ((sol, params, metrics, settings),)) 
                  for sol in sols]
        return [f.get() for f in futures]
    return batch_eval

def run_paes(icr_candidates, params, metrics, settings, cache_budget, archive_size, grid_divisions, max_evaluations, seed, allocs):
    logger.info("Run PAES")

    def mutate_fn_local(sol, rng):
        return mutate_fn(sol, rng)
    
    # n_proc = getattr(settings, 'N_PROCESSES', 2)  # fallback to 2
    n_proc = 10  # fallback to 2
    pool = Pool(processes=n_proc)
    
    def single_eval(sol):
        return eval_fn(sol, params=params, metrics=metrics, settings=settings)
    
    batch_eval = batch_eval_wrapper(pool, params, metrics, settings)

    init_fns_extra = [init_fn_alpha, init_fn_hub, init_fn_ci_only, init_fn_pr_only, init_fn_dc_only, init_fn_uniform, init_fn_one_shot, init_fn_top3]
    batch_init = batch_init_wrapper(pool, icr_candidates, params, cache_budget)
    
    opt = PAES(
        init_fn=init_fns_extra,
        mutate_fn=mutate_fn_local,
        eval_fn=single_eval,           # for single fallback
        batch_eval_fn=batch_eval,      # enables parallel batches
        batch_init_fn=batch_init,
        batch_size=9,                 # tune this
        archive_size=archive_size,
        grid_divisions=grid_divisions,
        max_evaluations=max_evaluations,
        seed=seed,
    )

    try:
        pareto = opt.run()
    finally:
        pool.close()
        pool.join()

    return pareto

# if __name__ == "__main__":
#     # Find Pareto solutions
#     ICR_CANDIDATES = [
#         4, 6, 10, 14, 15, 17, 18, 20, 21, 22,
#         29, 31, 34, 35, 36, 37, 38, 39, 40, 44,
#         45, 46, 49, 55, 56, 58, 59
#     ]
#     pareto = run_paes(icr_candidates=ICR_CANDIDATES,
#                     topology="GARR",
#                     alpha=1.2,
#                     strategy="CL2SM",
#                     network_cache=0.015,
#                     cache_placement="UNIFORM",
#                     archive_size=40,
#                     grid_divisions=30,
#                     max_evaluations=2,  # small for test, increase later
#                     seed=0)
    
#     print("Found", len(pareto), "Pareto solutions (max Hit, min Cost, min Carbon):")
#     for sol, (h, c, cf) in pareto:
#         print(sol["allocations"], " -> hit=%.6g, cost=%.6f, carbon=%.6f" % (-h, c, cf))
    
#     # Combined plot
#     hits = [-obj[0] for _, obj in pareto]  # invert -hit for plot
#     costs = [obj[1] for _, obj in pareto]
#     carbons = [obj[2] for _, obj in pareto]

    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(hits, costs, carbons, c="royalblue", s=60, label="PAES Pareto")

    # ax.set_xlabel("Hit Rate (maximize)")
    # ax.set_ylabel("Cost (minimize)")
    # ax.set_zlabel("Carbon (minimize)")
    # ax.set_title("PAES Pareto Front vs Baselines")
    # ax.legend()
    # plt.tight_layout()

    # out_path = EXAMPLES_DIR / "paes_logs/paes_vs_baselines_3d.png"
    # fig.savefig(out_path, dpi=300)
    # plt.close()
    # print(f"\n📌 3D plot saved to: {out_path}")
