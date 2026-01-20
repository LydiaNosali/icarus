import logging
import os
import pickle
from pathlib import Path
import copy
import random

from icarus.scenarios.paes.paes import PAES
from icarus.runner import run
import networkx as nx

EXAMPLES_DIR = Path(__file__).parent

logger = logging.getLogger("babel")

# ================== ci only init ========================
# def init_fn(icr_candidates, params, allocs, cache_budget):
#     centralities = params.get("network").get("node_carbon_intensity")
#     centralities = dict(centralities)
#     if not centralities:
#         raise ValueError("Carbon-aware init requires 'network.node_carbon_intensity'")

#     total_centrality = sum(centralities.values())

#     raw_alloc = {
#         v: cache_budget * centralities[v] / total_centrality
#         for v in centralities
#     }
    
#     # Initial rounding with minimum 1
#     rounded_alloc = {v: max(1, int(round(a))) for v, a in raw_alloc.items()}
#     total_allocated = sum(rounded_alloc.values())
#     while total_allocated > cache_budget:
#         # Find the node with the smallest allocation > 1 to reduce
#         over_nodes = [v for v in rounded_alloc if rounded_alloc[v] > 1]
#         if not over_nodes:
#             break  # Can't reduce anymore without violating ≥1 constraint
#         # Reduce the one with the smallest centrality
#         victim = min(over_nodes, key=lambda v: centralities[v])
#         rounded_alloc[victim] -= 1
#         total_allocated -= 1
    
#     allocs[:] = rounded_alloc.values()

#     # 8) Tier setup (unchanged from your pattern)
#     tiers_template = params.get("cache_policy", {}).get("tiers", {})
#     tiers_per_node = {node: copy.deepcopy(tiers_template) for node in icr_candidates}

#     return {
#         "allocations": allocs[:],
#         "cache_budget": cache_budget,
#         "actual_total": sum(rounded_alloc.values()),
#         "icr_candidates": icr_candidates,
#         "tiers_per_node": tiers_per_node,
#     }

# ================== centrality only init ========================
def init_fn(icr_candidates, params, allocs, cache_budget):
    net_params = params.get("network", {})
    topology = net_params.get("nx_graph")
    # betw = dict(nx.betweenness_centrality(topology))
    # pr_kwargs = {}
    # betw = dict(nx.pagerank(topology, **pr_kwargs))
    betw = dict(nx.degree(topology))

    centralities = {v: betw[v] for v in icr_candidates if betw[v] > 0}
    if not centralities:
        raise ValueError("No centralities")

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
    # print(allocs)

    # 8) Tier setup (unchanged from your pattern)
    tiers_template = params.get("cache_policy", {}).get("tiers", {})
    tiers_per_node = {node: copy.deepcopy(tiers_template) for node in icr_candidates}

    return {
        "allocations": allocs[:],
        "cache_budget": cache_budget,
        "actual_total": sum(rounded_alloc.values()),
        "icr_candidates": icr_candidates,
        "tiers_per_node": tiers_per_node,
    }

# ================== alpha * ci + (1 - alpha) * centrality ========================
# def init_fn(icr_candidates, params, allocs, cache_budget):
#     net_params = params.get("network", {})
#     topology = net_params.get("nx_graph")
#     ci = net_params.get("node_carbon_intensity")
#     if topology is None or ci is None:
#         raise ValueError("Init requires 'network.nx_graph' and 'network.node_carbon_intensity'")

#     # weight between CI and centrality
#     alpha = params.get("green", {}).get("alpha", 0.5)

#     # 1) topology centrality (degree here; swap to betweenness/pagerank if you like)
#     deg = dict(nx.betweenness_centrality(topology))

#     # restrict to candidates with positive degree
#     centralities = {v: float(deg[v]) for v in icr_candidates if deg.get(v, 0) > 0.0}
#     if not centralities:
#         raise ValueError("No centralities")

#     # 2) normalize centrality to [0,1]
#     cent_vals = list(centralities.values())
#     cent_min = min(cent_vals)
#     cent_max = max(cent_vals)
#     cent_range = cent_max - cent_min or 1.0
#     cent_norm = {v: (centralities[v] - cent_min) / cent_range for v in centralities}

#     # 3) normalize CI so that cleaner (lower CI) = higher score in [0,1]
#     ci_sub = {v: float(ci[v]) for v in centralities if v in ci}
#     if not ci_sub:
#         raise ValueError("No CI entries for the candidate nodes")
#     ci_vals = list(ci_sub.values())
#     ci_min = min(ci_vals)
#     ci_max = max(ci_vals)
#     ci_range = ci_max - ci_min or 1.0
#     # inverted & normalized: low CI → 1, high CI → 0
#     ci_norm = {v: (ci_max - ci_sub[v]) / ci_range for v in ci_sub}

#     # ensure all centrality nodes have a CI score (fallback 0 if missing)
#     for v in centralities:
#         ci_norm.setdefault(v, 0.0)

#     # 4) combined score: alpha * ci_norm + (1 - alpha) * cent_norm
#     scores = {
#         v: alpha * ci_norm[v] + (1.0 - alpha) * cent_norm[v]
#         for v in centralities
#     }

#     # drop any nodes with non-positive score
#     scores = {v: s for v, s in scores.items() if s > 0.0}
#     if not scores:
#         return {
#             "allocations": {},
#             "cache_budget": cache_budget,
#             "actual_total": 0,
#             "icr_candidates": icr_candidates,
#             "tiers_per_node": {},
#         }

#     total_score = sum(scores.values())

#     # 5) proportional allocation on combined score
#     raw_alloc = {
#         v: cache_budget * scores[v] / total_score
#         for v in scores
#     }

#     rounded_alloc = {v: max(1, int(round(a))) for v, a in raw_alloc.items()}
#     total_allocated = sum(rounded_alloc.values())

#     # if we overshoot the budget, decrement lowest-score nodes first
#     while total_allocated > cache_budget:
#         over_nodes = [v for v in rounded_alloc if rounded_alloc[v] > 1]
#         if not over_nodes:
#             break
#         victim = min(over_nodes, key=lambda v: scores[v])
#         rounded_alloc[victim] -= 1
#         total_allocated -= 1

#     # if we undershoot (because of rounding), give leftover to highest-score nodes
#     while total_allocated < cache_budget:
#         beneficiary = max(rounded_alloc.keys(), key=lambda v: scores[v])
#         rounded_alloc[beneficiary] += 1
#         total_allocated += 1

#     # map allocs[] (which is ordered like icr_candidates) from the dict
#     alloc_map = {v: 0 for v in icr_candidates}
#     alloc_map.update(rounded_alloc)
#     allocs[:] = [alloc_map[v] for v in icr_candidates]

#     tiers_template = params.get("cache_policy", {}).get("tiers", {})
#     tiers_per_node = {node: copy.deepcopy(tiers_template) for node in icr_candidates}

#     return {
#         "allocations": allocs[:],
#         "cache_budget": cache_budget,
#         "actual_total": total_allocated,
#         "icr_candidates": icr_candidates,
#         "tiers_per_node": tiers_per_node,
#     }

# ================== hub fraction ========================
# def init_fn(icr_candidates, params, allocs, cache_budget):
#     net_params = params.get("network", {})
#     topology = net_params.get("nx_graph")
#     ci = net_params.get("node_carbon_intensity")
#     if topology is None or ci is None:
#         raise ValueError("Init requires 'network.nx_graph' and 'network.node_carbon_intensity'")
    
#     # Tunable parameters
#     hub_fraction = params.get("network", {}).get("hub_fraction")  # % of nodes as hubs
#     hub_share = params.get("network", {}).get("hub_budget_share")  # Budget % to hubs

#     deg = dict(nx.betweenness_centrality(topology))  # degree centrality related to importance [web:27]
#     centralities = {v: float(deg.get(v, 0.0)) for v in icr_candidates}
#     # filter positive centrality
#     centralities = {v: c for v, c in centralities.items() if c > 0.0}
    
#     if not centralities:
#         raise ValueError("No centralities")
    
#     nodes = list(centralities.keys())
#     n = len(nodes)

#     k = max(1, int(round(hub_fraction * n)))
#     # sort by centrality descending
#     nodes_sorted = sorted(nodes, key=lambda v: centralities[v], reverse=True)
#     hub_nodes = set(nodes_sorted[:k])
#     non_hub_nodes = set(nodes) - hub_nodes

#     hub_budget = cache_budget * hub_share
#     non_hub_budget = cache_budget - hub_budget

#     if hub_nodes:
#         hub_cents = {v: centralities[v] for v in hub_nodes}
#         total_hub_cent = sum(hub_cents.values())
#         # if degenerate, fall back to uniform among hubs
#         if total_hub_cent <= 0:
#             hub_raw = {v: hub_budget / len(hub_nodes) for v in hub_nodes}
#         else:
#             hub_raw = {v: hub_budget * hub_cents[v] / total_hub_cent for v in hub_nodes}
#     else:
#         hub_raw = {}

#     # Phase 2: Allocate remainder to non-hubs using CI only
#     if non_hub_nodes:
#         ci_non = {v: float(ci[v]) for v in non_hub_nodes if v in ci}
#         if not ci_non:
#             # if no CI info, uniform among non-hubs
#             non_hub_raw = {v: non_hub_budget / len(non_hub_nodes) for v in non_hub_nodes}
#         else:
#             ci_vals = list(ci_non.values())
#             ci_min = min(ci_vals)
#             ci_max = max(ci_vals)
#             ci_range = ci_max - ci_min or 1.0
#             # inverted & normalized: low CI → high score
#             ci_score = {v: (ci_max - ci_non[v]) / ci_range for v in ci_non}
#             # ensure all non-hubs have some score (0 if missing)
#             for v in non_hub_nodes:
#                 ci_score.setdefault(v, 0.0)
#             total_ci_score = sum(ci_score.values())
#             if total_ci_score <= 0:
#                 non_hub_raw = {v: non_hub_budget / len(non_hub_nodes) for v in non_hub_nodes}
#             else:
#                 non_hub_raw = {
#                     v: non_hub_budget * ci_score[v] / total_ci_score
#                     for v in non_hub_nodes
#                 }
#     else:
#         non_hub_raw = {}
    
#     # 6) merge raw allocations and round, enforcing ≥1 for any node that gets non-zero
#     raw_alloc = {}
#     raw_alloc.update(hub_raw)
#     raw_alloc.update(non_hub_raw)

#     # only keep nodes with positive raw allocation
#     raw_alloc = {v: a for v, a in raw_alloc.items() if a > 0.0}

#     if not raw_alloc:
#         raise ValueError("No raw_alloc")

#     rounded_alloc = {v: max(1, int(round(a))) for v, a in raw_alloc.items()}
#     total_allocated = sum(rounded_alloc.values())

#     # 7) adjust down if over budget: remove from lowest-priority nodes
#     # priority: non-hubs first (penalize greener fringe before hubs), then lowest centrality
#     def priority_key(v):
#         # higher = better; we want to remove from lowest
#         is_hub = 1 if v in hub_nodes else 0
#         return (is_hub, centralities.get(v, 0.0))
   
#     while total_allocated > cache_budget:
#         over_nodes = [v for v in rounded_alloc if rounded_alloc[v] > 1]
#         if not over_nodes:
#             break
#         victim = min(over_nodes, key=priority_key)  # lowest priority
#         rounded_alloc[victim] -= 1
#         total_allocated -= 1

#     # 8) if under budget (rare), add to highest priority nodes (hubs with high centrality)
#     while total_allocated < cache_budget:
#         beneficiary = max(rounded_alloc.keys(), key=priority_key)
#         rounded_alloc[beneficiary] += 1
#         total_allocated += 1

#     # 9) map back to allocs[] order (icr_candidates)
#     alloc_map = {v: 0 for v in icr_candidates}
#     alloc_map.update(rounded_alloc)
#     allocs[:] = [alloc_map[v] for v in icr_candidates]

#     tiers_template = params.get("cache_policy", {}).get("tiers", {})
#     tiers_per_node = {node: copy.deepcopy(tiers_template) for node in icr_candidates}
    
#     return {
#         "allocations": allocs[:],
#         "cache_budget": cache_budget,
#         "actual_total": total_allocated,
#         "icr_candidates": icr_candidates,
#         "tiers_per_node": tiers_per_node,
#     }

# --- Mutation function ---
def mutate_fn(sol, rng):
    new_sol = copy.deepcopy(sol)
    alloc_list = new_sol["allocations"][:]
    n_nodes = len(alloc_list)
    budget = sol["cache_budget"]

    move_fraction = rng.uniform(0.1, 0.25)  # 10-25% budget
    amount_to_move = int(budget * move_fraction)  # 20-50 units!
    
    # 1) Find non-zero nodes (always safe)
    non_zero = [i for i in range(n_nodes) if alloc_list[i] > 0]
    n_non_zero = len(non_zero)

    if n_non_zero >= 2 and amount_to_move > 0:
        # STEAL: Take 1-3 from 3-5 donors (SIMPLE arithmetic, NO randint)
        # n_donors = min(5, len(non_zero))
        # donors = non_zero[:n_donors]
        total_donor_alloc = sum(alloc_list[i] for i in non_zero)
        donor_probs = [alloc_list[i] / total_donor_alloc for i in non_zero]
        donors = rng.choices(non_zero, weights=donor_probs, k=3)

        # donor_pct = rng.uniform(0.2, 0.5) 
        # n_donors = max(3, min(int(n_non_zero * donor_pct), n_non_zero // 2))
        # donors = rng.sample(non_zero, n_donors) # random subset
        # donors = non_zero[:n_donors]
        # print(f"donora:{donors}")

        for donor in donors:
            steal = min(amount_to_move//3 + 1, alloc_list[donor]//2)
            alloc_list[donor] = max(0, alloc_list[donor] - steal)
            amount_to_move -= steal
            # if alloc_list[donor] >= 4:  # Safe threshold
            #     steal = min(3, alloc_list[donor]//4 + 1)  # 1-3, NO randint
            #     alloc_list[donor] -= steal

        # 2) GIVE to 6-10 random nodes
        # n_receivers = rng.randint(6, min(10, n_nodes))
        
        # receiver_pct = rng.uniform(0.2, 0.5) 
        # n_receivers = max(6, int(n_nodes * receiver_pct))
        # n_receivers = min(n_receivers, n_nodes)
        
        # receivers = rng.sample(range(n_nodes), n_receivers)
        # print(f"receivers:{receivers}")
        # total_stolen = sum(max(0, min(3, alloc_list[i]//4 + 1)) for i in donors)
        # chunk = max(1, total_stolen // n_receivers)
        # Give to diverse receivers (favor zero nodes for exploration)
        zero_nodes = [i for i in range(n_nodes) if alloc_list[i] == 0]
        non_zero_nodes = [i for i in non_zero if alloc_list[i] > 0]
        
        if zero_nodes:
            receivers = (rng.sample(zero_nodes, min(2, len(zero_nodes))) + 
                        rng.sample(non_zero_nodes, min(3, len(non_zero_nodes))))
        else:
            receivers = rng.sample(range(n_nodes), 5)
        for receiver in receivers:
            give = min(amount_to_move // len(receivers) + 1, amount_to_move)
            alloc_list[receiver] += give
            amount_to_move -= give
            # alloc_list[receiver] += chunk
    
    # Floor + exact budget (bulletproof)
    # for i in range(n_nodes):
    #     alloc_list[i] = max(0, alloc_list[i])
    
    # total = sum(alloc_list)
    # while total > budget:
    #     donor = next((i for i in range(n_nodes) if alloc_list[i] > 0), None)
    #     if donor is not None:
    #         alloc_list[donor] -= 1
    #         total -= 1
    # while total < budget:
    #     idx = rng.choice(range(n_nodes))
    #     alloc_list[idx] += 1
    #     total += 1
    
    # # Convert back
    # if hasattr(new_sol["allocations"], 'get'):
    #     new_sol["allocations"] = {i: v for i, v in enumerate(alloc_list) if v > 0}
    # else:
    #     new_sol["allocations"] = alloc_list
    # Exact budget snapback
    total = sum(alloc_list)
    while total > budget: alloc_list[rng.choice(non_zero)] -= 1; total -= 1
    while total < budget: alloc_list[rng.choice(range(n_nodes))] += 1; total += 1
    
    new_sol["allocations"] = [max(0, x) for x in alloc_list]

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
    # exp["workload"]["n_warmup"] /= 5
    # exp["workload"]["n_measured"] /= 2
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
        return (hit, cost, carbon)
    except Exception as e:
        print("Error reading results:", e)
        return (None, None)


def run_paes(icr_candidates, params, metrics, settings, cache_budget, archive_size, grid_divisions, max_evaluations, seed, allocs):
    logger.info("Run PAES")
    # Wrap mutate_fn to use the local RNG
    def mutate_fn_local(sol, rng):
        rng = random.Random(seed)
        return mutate_fn(sol, rng)
    opt = PAES(
        init_fn=lambda: init_fn(icr_candidates, params, allocs, cache_budget),
        mutate_fn=mutate_fn_local,
        eval_fn=lambda sol: eval_fn(sol, params=params, metrics=metrics, settings=settings),
        archive_size=archive_size,
        grid_divisions=grid_divisions,
        max_evaluations=max_evaluations,
        seed=seed,
    )
    pareto = opt.run()

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
