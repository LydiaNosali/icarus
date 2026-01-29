# icarus/scenarios/nsga2/nsga2.py
from __future__ import annotations

import copy
import logging
import networkx as nx
import random
import os
import pickle
from pathlib import Path
from icarus.runner import run
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from icarus.util import iround

EXAMPLES_DIR = Path(__file__).parent

logger = logging.getLogger("babel")

Objectives = Tuple[float, float, float]   # (carbon, hit, cost)
Solution = Dict[str, Any]

# -----------------------------
# Dominance with mixed directions
# directions: +1 maximize, -1 minimize
# Here: hit maximize, cost minimize, carbon minimize
# -----------------------------
DIRECTIONS = (-1, +1, -1)

# ================== base init ========================
def init_fn_uniform(icr_candidates, params, allocs, cache_budget):
    cache_size = iround(cache_budget / len(icr_candidates))
    
    raw_alloc = {
        v: cache_size
        for v in icr_candidates
    }
    
    # Initial rounding with minimum 1
    rounded_alloc = {v: max(1, int(round(a))) for v, a in raw_alloc.items()}
    
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
    
    # Initial rounding with minimum 1
    rounded_alloc = {v: max(1, int(round(a))) for v, a in raw_alloc.items()}
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

# ================== bc only init ========================
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
    
    # Initial rounding with minimum 1
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

# ================== pr only init ========================
def init_fn_pr_only(icr_candidates, params, allocs, cache_budget):
    pr_kwargs = {}
    net_params = params.get("network", {})
    topology = net_params.get("nx_graph")
    # betw = dict(nx.betweenness_centrality(topology))
    # pr_kwargs = {}
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
    
    # Initial rounding with minimum 1
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

    rounded_alloc = {v: max(1, int(round(a))) for v, a in raw_alloc.items()}
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

    rounded_alloc = {v: max(1, int(round(a))) for v, a in raw_alloc.items()}
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

def mutate_fn(sol, rng):
    # ==========================================================
    # MODE 0 — STRUCTURAL RESET (rare, but critical)
    # ==========================================================
    new_sol = copy.deepcopy(sol)
    alloc = new_sol["allocations"][:]
    n = len(alloc)
    nodes = list(range(n))
    budget = new_sol["cache_budget"]
    icr = new_sol["icr_candidates"]
    icr = list(icr)
    if rng.random() < 0.07:   # 7% probability
        alloc = [0] * n

        # choose a random subset of nodes (not necessarily central)
        k = rng.randint(3, max(3, n // 3))
        chosen = rng.sample(nodes, k)

        # redistribute entire budget randomly
        for _ in range(int(budget)):
            alloc[rng.choice(chosen)] += 1

        new_sol["allocations"] = alloc
        return new_sol
    
    # ---- parameters ----
    modes = ["carbon"]*6 + ["centrality"]*2 + ["random"]*1  
    mode = rng.choice(modes)
    print(f"{mode} MUTATE")

    # mode-dependent mutation strength
    if mode == "random":
        move_frac = rng.uniform(0.15, 0.35)   # strong exploration
    elif mode == "carbon":
        move_frac = rng.uniform(0.08, 0.2)
    else:  # hit
        move_frac = rng.uniform(0.05, 0.15)

    amount = max(1, int(budget * move_frac))

    # ---- helpers ----
    non_zero = [i for i in nodes if alloc[i] > 0]
    if len(non_zero) < 2:
        return new_sol

    # ==========================================================
    # MODE 1 — CARBON-AWARE (push cache toward clean nodes)
    # ==========================================================
    if mode == "carbon":
        ci = new_sol["node_carbon_intensity"]
        ranked = sorted(non_zero, key=lambda i: ci.get(icr[i]))
        donors = ranked[-max(2, len(ranked)//4):]     # dirty nodes
        receivers = ranked[:max(2, len(ranked)//4)]   # clean nodes

    # ==========================================================
    # MODE 2 — HIT-ORIENTED (reinforce strong caches)
    # ==========================================================
    elif mode == "centrality":
        centralities = new_sol["centralities"]
        ranked = sorted(non_zero, key=lambda i: centralities.get(icr[i]))
        donors = ranked[:max(2, len(ranked)//4)]      # low-central nodes
        receivers = ranked[-max(2, len(ranked)//4):]  # high-central nodes

    # ==========================================================
    # MODE 3 — RANDOM / SHAKE (escape local basin)
    # ==========================================================
    else:
        # occasionally kill a node entirely
        if rng.random() < 0.3:
            victim = rng.choice(non_zero)
            amount += alloc[victim]
            alloc[victim] = 0
            non_zero.remove(victim)

        donors = rng.sample(non_zero, min(5, len(non_zero)))
        receivers = rng.sample(nodes, rng.randint(4, n))

    # ---- MOVE CACHE MASS ----
    # steal
    for d in donors:
        if amount <= 0:
            break
        steal = min(alloc[d], max(1, amount // len(donors)))
        alloc[d] -= steal
        amount -= steal

    # give
    for r in receivers:
        if amount <= 0:
            break
        give = max(1, amount // len(receivers))
        alloc[r] += give
        amount -= give

    # ---- SNAP BUDGET EXACTLY ----
    total = sum(alloc)
    while total > budget:
        i = rng.choice([i for i in nodes if alloc[i] > 0])
        alloc[i] -= 1
        total -= 1
    while total < budget:
        alloc[rng.choice(nodes)] += 1
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
    exp["workload"]["n_warmup"] /= 5
    exp["workload"]["n_measured"] /= 5
    exp["workload"]["n_measured"] = int(exp["workload"]["n_measured"])
     # tiers logic unchanged
    if tiers_per_node is not None:
        exp["cache_policy"]["tiers_per_node"] = tiers_per_node
    
    return exp

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
        logger.info(f"hit : {hit}, cost : {cost}, carbon : {carbon}")
        print(f"hit : {hit}, cost : {cost}, carbon : {carbon}")

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
        return (None, None)


def _dominates(a: Objectives, b: Objectives, directions=DIRECTIONS) -> bool:
    """
    True if 'a' Pareto-dominates 'b' with mixed objective directions.
    """
    better_or_equal = True
    strictly_better = False

    for (va, vb, d) in zip(a, b, directions):
        if d == +1:  # maximize
            if va < vb:
                better_or_equal = False
                break
            if va > vb:
                strictly_better = True
        else:        # minimize
            if va > vb:
                better_or_equal = False
                break
            if va < vb:
                strictly_better = True

    return better_or_equal and strictly_better


def _key(sol: Solution) -> Tuple[int, ...]:
    return tuple(sol["allocations"])


@dataclass
class _Individual:
    sol: Solution
    obj: Objectives
    rank: int = 10**9
    crowd: float = 0.0


class NSGA2:
    """
    Minimal NSGA-II specialized for your solution shape:
      - sol["allocations"] : list[int]
      - objectives: (carbon, hit, cost) with mixed directions
    """

    def __init__(
        self,
        init_solutions: List[Callable[[], Solution]],
        mutate: Callable[[Solution, random.Random], Solution],
        evaluate: Callable[[Solution], Objectives],
        pop_size: int = 40,
        max_evaluations: int = 200,
        seed: Optional[int] = 0,
        crossover_prob: float = 0.9,
        mutation_prob: float = 1.0,
        tournament_k: int = 2,
    ):
        self.rng = random.Random(seed)
        self.init_solutions = init_solutions
        self.mutate = mutate
        self.evaluate = evaluate

        self.pop_size = pop_size
        self.max_evaluations = max_evaluations

        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_k = tournament_k

        # cache evaluations (very important since eval runs Icarus)
        self._eval_cache: Dict[Tuple[int, ...], Objectives] = {}

        self._eval_count = 0

    def _get_or_eval(self, sol: Solution) -> Objectives:
        k = _key(sol)
        if k in self._eval_cache:
            return self._eval_cache[k]

        obj = self.evaluate(sol)
        self._eval_cache[k] = obj
        self._eval_count += 1
        return obj

    # -----------------------------
    # Fast non-dominated sorting
    # -----------------------------
    def _fast_nondominated_sort(self, pop: List[_Individual]) -> List[List[_Individual]]:
        S: Dict[int, List[int]] = {}
        n: Dict[int, int] = {}
        fronts: List[List[int]] = [[]]

        for p_i, p in enumerate(pop):
            S[p_i] = []
            n[p_i] = 0
            for q_i, q in enumerate(pop):
                if p_i == q_i:
                    continue
                if _dominates(p.obj, q.obj):
                    S[p_i].append(q_i)
                elif _dominates(q.obj, p.obj):
                    n[p_i] += 1

            if n[p_i] == 0:
                p.rank = 0
                fronts[0].append(p_i)

        i = 0
        while fronts[i]:
            next_front: List[int] = []
            for p_i in fronts[i]:
                for q_i in S[p_i]:
                    n[q_i] -= 1
                    if n[q_i] == 0:
                        pop[q_i].rank = i + 1
                        next_front.append(q_i)
            i += 1
            fronts.append(next_front)

        # convert indices → individuals and drop last empty
        ind_fronts: List[List[_Individual]] = []
        for f in fronts:
            if not f:
                break
            ind_fronts.append([pop[idx] for idx in f])
        return ind_fronts

    # -----------------------------
    # Crowding distance
    # -----------------------------
    def _crowding_distance(self, front: List[_Individual]) -> None:
        if len(front) <= 2:
            for ind in front:
                ind.crowd = float("inf")
            return
        
        n_obj = 3  # carbon, hit, cost
        
        # Initialize crowding distances
        for ind in front:
            ind.crowd = 0.0
        
        # For each objective: sort, normalize, compute distances
        for j in range(n_obj):
            # Sort front by objective j
            front.sort(key=lambda x: x.obj[j])
            front[0].crowd = float("inf")   # boundary points
            front[-1].crowd = float("inf")
            
            # Get range of THIS objective in THIS front for normalization
            f_min = front[0].obj[j]
            f_max = front[-1].obj[j]
            f_range = f_max - f_min if f_max != f_min else 1.0
            
            # Compute normalized crowding for interior points
            for i in range(1, len(front) - 1):
                f_prev = front[i-1].obj[j]
                f_next = front[i+1].obj[j]
                front[i].crowd += (f_next - f_prev) / f_range
        
        # CARBON BIAS: slightly favor low-carbon solutions
        for ind in front:
            ind.crowd *= (1.0 + 1.0 / (1.0 + ind.obj[0]))  # lower carbon → bonus

    # -----------------------------
    # Tournament selection (rank, then crowding)
    # -----------------------------
    def _tournament(self, pop: List[_Individual]) -> _Individual:
        contenders = [pop[self.rng.randrange(len(pop))] for _ in range(self.tournament_k)]
        # rank first, then crowding, then LOWEST carbon breaks ties
        contenders.sort(key=lambda ind: (ind.rank, -ind.crowd, ind.obj[0]))
        return contenders[0]

    # -----------------------------
    # Crossover for integer allocations + repair budget
    # -----------------------------
    def _crossover(self, a: Solution, b: Solution) -> Tuple[Solution, Solution]:
        # Uniform crossover on allocations; keep other fields from parents (same anyway)
        ca = copy.deepcopy(a)
        cb = copy.deepcopy(b)

        alloc_a = ca["allocations"][:]
        alloc_b = cb["allocations"][:]
        n = len(alloc_a)

        for i in range(n):
            if self.rng.random() < 0.5:
                alloc_a[i], alloc_b[i] = alloc_b[i], alloc_a[i]

        ca["allocations"] = self._repair_budget(ca, alloc_a)
        cb["allocations"] = self._repair_budget(cb, alloc_b)
        return ca, cb

    def _repair_budget(self, template_sol: Solution, alloc: List[int]) -> List[int]:
        # Ensures all >=0 and sum == cache_budget (exactly),
        # same constraint your mutate_fn enforces. :contentReference[oaicite:4]{index=4}
        alloc = [max(0, int(x)) for x in alloc]
        budget = int(template_sol["cache_budget"])
        n = len(alloc)
        total = sum(alloc)

        if n == 0:
            return alloc

        # If everything is zero but budget > 0, seed something
        if total == 0 and budget > 0:
            alloc[self.rng.randrange(n)] = budget
            return alloc

        # Normalize to budget by random +/-1 moves
        while total > budget:
            candidates = [i for i in range(n) if alloc[i] > 0]
            if not candidates:
                break
            i = self.rng.choice(candidates)
            alloc[i] -= 1
            total -= 1

        while total < budget:
            i = self.rng.randrange(n)
            alloc[i] += 1
            total += 1

        return alloc

    # -----------------------------
    # Main run
    # -----------------------------
    def run(self) -> List[Tuple[Solution, Objectives]]:
        # ---- 1) build initial population (seed with your extreme heuristics)
        pop: List[_Individual] = []

        # Ensure we have enough seed generators to fill pop
        seed_fns = self.init_solutions[:]
        if not seed_fns:
            raise ValueError("NSGA2 requires at least one init solution generator.")

        seen = set()
        while len(pop) < self.pop_size and self._eval_count < self.max_evaluations:
            fn = seed_fns[len(pop) % len(seed_fns)]
            sol = fn()

            # If duplicates, perturb a bit
            k = _key(sol)
            if k in seen:
                sol = self.mutate(sol, self.rng)
                k = _key(sol)
                if k in seen:
                    continue

            obj = self._get_or_eval(sol)
            if obj[0] is None or obj[1] is None or obj[2] is None:
                continue

            pop.append(_Individual(sol=sol, obj=obj))
            seen.add(k)

        # If still short (e.g. due to failed evals), keep mutating last valid
        while len(pop) < self.pop_size and self._eval_count < self.max_evaluations and pop:
            base = pop[-1].sol
            sol = self.mutate(base, self.rng)
            k = _key(sol)
            if k in seen:
                continue
            obj = self._get_or_eval(sol)
            if obj[0] is None or obj[1] is None or obj[2] is None:
                continue
            pop.append(_Individual(sol=sol, obj=obj))
            seen.add(k)

        # ---- 2) evolve until we hit max_evaluations
        while self._eval_count < self.max_evaluations:
            # Rank + crowding for selection
            fronts = self._fast_nondominated_sort(pop)
            for f in fronts:
                self._crowding_distance(f)

            # Create offspring
            offspring: List[_Individual] = []
            while len(offspring) < self.pop_size and self._eval_count < self.max_evaluations:
                p1 = self._tournament(pop)
                p2 = self._tournament(pop)

                c1_sol = copy.deepcopy(p1.sol)
                c2_sol = copy.deepcopy(p2.sol)

                if self.rng.random() < self.crossover_prob:
                    c1_sol, c2_sol = self._crossover(p1.sol, p2.sol)

                if self.rng.random() < self.mutation_prob:
                    c1_sol = self.mutate(c1_sol, self.rng)
                if self.rng.random() < self.mutation_prob:
                    c2_sol = self.mutate(c2_sol, self.rng)

                for child_sol in (c1_sol, c2_sol):
                    k = _key(child_sol)
                    if k in seen:
                        continue
                    obj = self._get_or_eval(child_sol)
                    if obj[0] is None or obj[1] is None or obj[2] is None:
                        continue
                    offspring.append(_Individual(sol=child_sol, obj=obj))
                    seen.add(k)
                    if len(offspring) >= self.pop_size:
                        break

            # Combine and select next generation
            combined = pop + offspring
            fronts = self._fast_nondominated_sort(combined)

            next_pop: List[_Individual] = []
            for f in fronts:
                self._crowding_distance(f)
                if len(next_pop) + len(f) <= self.pop_size:
                    next_pop.extend(f)
                else:
                    # take most diverse
                    f.sort(key=lambda ind: -ind.crowd)
                    next_pop.extend(f[: self.pop_size - len(next_pop)])
                    break

            pop = next_pop

            # If we couldn’t create new individuals, break (avoid infinite loops)
            if not offspring:
                break

        # ---- 3) return final non-dominated set (front 0) as (sol, obj)
        fronts = self._fast_nondominated_sort(pop)
        pareto_front = fronts[0] if fronts else []
        return [(ind.sol, ind.obj) for ind in pareto_front]


# -------------------------------------------------------
# Public entry point called by cacheplacement.py
# -------------------------------------------------------
def run_nsga2(
        icr_candidates: List[int],
        params: Dict[str, Any],
        metrics: Any,
        settings: Any,
        cache_budget: int,
        archive_size: int = 40,
        max_evaluations: int = 120,
        seed: int = 0,
    ) -> List[Tuple[Solution, Objectives]]:
    """
    Mirrors your old run_paes(...) signature but runs NSGA-II.

    Returns: List[(sol_dict, (carbon, hit, cost))] to match what your
    green_cache_placement expects. :contentReference[oaicite:5]{index=5}
    """
    def init_uniform() -> Solution:
        return init_fn_uniform(icr_candidates, params, allocs=[], cache_budget=cache_budget)
    
    def init_pr() -> Solution:
        return init_fn_pr_only(icr_candidates, params, allocs=[], cache_budget=cache_budget)
    
    def init_bc() -> Solution:
        return init_fn_bc_only(icr_candidates, params, allocs=[], cache_budget=cache_budget)

    def init_carbon() -> Solution:
        return init_fn_ci_only(icr_candidates, params, allocs=[], cache_budget=cache_budget)

    def init_alpha() -> Solution:
        return init_fn_alpha(icr_candidates, params, allocs=[], cache_budget=cache_budget)

    def init_hub() -> Solution:
        return init_fn_hub(icr_candidates, params, allocs=[], cache_budget=cache_budget)

    def mutate_local(sol: Solution, rng: random.Random) -> Solution:
        return mutate_fn(sol, rng)

    def eval_local(sol: Solution) -> Objectives:
        return eval_fn(sol, params=params, metrics=metrics, settings=settings)

    pop_size = int(archive_size)  # use archive_size as population size (simple mapping)

    nsga2 = NSGA2(
        init_solutions=[init_uniform, init_bc, init_pr, init_carbon, init_alpha, init_hub],
        mutate=mutate_local,
        evaluate=eval_local,
        pop_size=pop_size,
        max_evaluations=int(max_evaluations),
        seed=seed,
        crossover_prob=0.9,
        mutation_prob=1.0,
        tournament_k=2,
    )
    pareto_solutions = nsga2.run()
    
    carbon_first = sorted(pareto_solutions, key=lambda x: x[1][0])
    
    print(f"✅ Top 3 green: carbon={carbon_first[0][1][0]:.1f}, hit={carbon_first[0][1][1]:.3f}")
    return carbon_first
