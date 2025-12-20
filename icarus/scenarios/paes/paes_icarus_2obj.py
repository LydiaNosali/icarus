import logging
import pickle
from pathlib import Path
import copy
import math
import random
import statistics

from matplotlib import pyplot as plt
import numpy as np
from icarus.scenarios.paes.paes import PAES
from icarus.runner import run
from icarus.util import iround

EXAMPLES_DIR = Path(__file__).parent

logger = logging.getLogger("babel")

def init_fn(icr_candidates, params, allocs, cache_budget):
    logger.info("Init carbon & betweenness aware (tunable importance)")

    ci = params.get("network", {}).get("node_carbon_intensity")
    bc = params.get("network", {}).get("node_betweenness")

    if not ci:
        raise ValueError("Carbon-aware init requires 'network.node_carbon_intensity'")
    if not bc:
        raise ValueError("Betweenness-aware init requires 'network.node_betweenness'")

    # Tunable importance factor: alpha = importance of CI (0 = only BC, 1 = only CI)
    alpha = params.get("init", {}).get("ci_weight", 0.2)  # default 0.5 = equal importance

    # Normalize CI (inverted: lower carbon = higher normalized value)
    ci_values = [ci.get(node, 1.0) for node in icr_candidates]
    max_ci, min_ci = max(ci_values), min(ci_values)
    ci_norm = [(max_ci - val) / (max_ci - min_ci + 1e-9) for val in ci_values]

    # Normalize BC
    bc_values = [bc.get(node, 0.0) for node in icr_candidates]
    total_bc = sum(bc_values) or 1.0
    bc_norm = [v / total_bc for v in bc_values]

    # Combine weights with tunable importance
    weights = [alpha * c + (1 - alpha) * b for c, b in zip(ci_norm, bc_norm)]
    total_w = sum(weights) or 1.0

    # Allocation proportional to weights
    raw = [cache_budget * w / total_w for w in weights]
    floored = [int(math.floor(a)) for a in raw]
    remainder = int(cache_budget - sum(floored))

    # Distribute remaining cache to nodes with largest fractional parts
    fracs = sorted(
        enumerate([a - f for a, f in zip(raw, floored)]),
        key=lambda x: x[1],
        reverse=True,
    )
    for i, _ in fracs[:remainder]:
        floored[i] += 1

    allocs = floored

    tiers_template = params["cache_policy"]["tiers"]
    tiers_per_node = {node: copy.deepcopy(tiers_template) for node in icr_candidates}

    sol = {
        "allocations": allocs,
        "cache_budget": cache_budget,
        "icr_candidates": icr_candidates,
        "tiers_per_node": tiers_per_node,
        "node_carbon_intensity": ci,
    }
    return sol

# def init_fn(icr_candidates, params, allocs):
#     logger.info("Init with Uniform / Green-aware")
#     cache_budget = params["workload"]["n_contents"] * params["cache_placement"]["network_cache"]
#     # # Fallback: pure uniform baseline
#     # logger.info("No CI information, using pure uniform seeding")
#     # cache_size = iround(cache_budget / len(icr_candidates))
#     # allocs = [cache_size] * len(icr_candidates)

#     ci = params.get("network", {}).get("node_carbon_intensity")
#     # CASE 1: external allocations provided -> just use them
#     if allocs:
#         logger.info(f"In init, external allocs: {allocs}")
#     else:
#         # CASE 2: no allocs → build them here
#         # if ci:
#         #     # GREEN: carbon-aware seeding
#         #     logger.info("Carbon-aware seeding based on node_carbon_intensity")
#         #     # lower CI => higher weight
#         #     weights = []
#         #     for node in icr_candidates:
#         #         val = ci.get(node, 1.0)
#         #         weights.append(1.0 / (val + 1e-9))

#         #     total_w = sum(weights) or 1.0
#         #     # proportional allocation
#         #     raw = [cache_budget * w / total_w for w in weights]
#         #     floored = [int(math.floor(a)) for a in raw]
#         #     remainder = int(cache_budget - sum(floored))

#         #     # give remainders to largest fractional parts
#         #     fracs = sorted(
#         #         enumerate([a - f for a, f in zip(raw, floored)]),
#         #         key=lambda x: x[1],
#         #         reverse=True,
#         #     )
#         #     for i, _ in fracs[:remainder]:
#         #         floored[i] += 1

#         #     allocs = floored
#         # else:
#         # Fallback: pure uniform baseline
#         logger.info("No CI information, using pure uniform seeding")
#         cache_size = iround(cache_budget / len(icr_candidates))
#         allocs = [cache_size] * len(icr_candidates)

#     logger.info(f"In init, final allocs: {allocs}")
#     tiers_per_node = {node: copy.deepcopy(params["cache_policy"]["tiers"]) for node in icr_candidates}

#     sol = {
#         "allocations": allocs,
#         "cache_budget": cache_budget,
#         "icr_candidates": icr_candidates,
#         "tiers_per_node": tiers_per_node,
#         "node_carbon_intensity": ci,
#     }
#     return sol

# def init_fn(icr_candidates, params, allocs, cache_budget):
#     logger.info("Init 100% carbon-aware")

#     ci = params.get("network", {}).get("node_carbon_intensity")

#     # 1) Si des allocations externes sont fournies, on les respecte
#     # if allocs:
#     #     logger.info(f"In init, external allocs: {allocs}")
#     # else:
#     if not ci:
#         raise ValueError(
#             "Carbon-aware init requires 'network.node_carbon_intensity'"
#         )

#     # Poids = inverse de l'intensité carbone (plus c'est vert, plus on alloue)
#     weights = []
#     for node in icr_candidates:
#         val = ci.get(node, 1.0)  # défaut = sale
#         weights.append(1.0 / (val + 1e-9))

#     total_w = sum(weights) or 1.0

#     # Allocation proportionnelle aux poids carbone
#     raw = [cache_budget * w / total_w for w in weights]
#     floored = [int(math.floor(a)) for a in raw]
#     remainder = int(cache_budget - sum(floored))

#     # Donner les restes aux plus grosses parties fractionnaires
#     fracs = sorted(
#         enumerate([a - f for a, f in zip(raw, floored)]),
#         key=lambda x: x[1],
#         reverse=True,
#     )
#     for i, _ in fracs[:remainder]:
#         floored[i] += 1

#     allocs = floored

#     tiers_template = params["cache_policy"]["tiers"]
#     tiers_per_node = {
#         node: copy.deepcopy(tiers_template) for node in icr_candidates
#     }

#     sol = {
#         "allocations": allocs,
#         "cache_budget": cache_budget,
#         "icr_candidates": icr_candidates,
#         "tiers_per_node": tiers_per_node,
#         "node_carbon_intensity": ci,
#     }
#     return sol

# import copy
# import numpy as np

# def init_fn(icr_candidates, params, allocs=None):
#     """
#     Init that seeds cache where it most reduces traffic CF:
#     - high traffic (demand / server hits)
#     - high betweenness (on many paths)
#     - low CI (tie-breaker)
#     """
#     logger.info("Init with traffic + BC + CI seeding")
#     cache_budget = int(round(params["workload"]["n_contents"] *
#                              params["cache_placement"]["network_cache"]))

#     # Ensure deterministic order
#     nodes = list(icr_candidates)

#     net = params.get("network", {})
#     # traffic = net.get("node_traffic", {})            # node -> traffic weight
#     # bc = net.get("node_betweenness", {})             # node -> betweenness
#     ci = net.get("node_carbon_intensity", {})        # node -> CI

#     n_nodes = len(nodes)

#     if allocs:
#         logger.info(f"In init, external allocs: {allocs}")
#     else:
#         # gather metrics
#         # traffic_vals = np.array([traffic.get(node, 0.0) for node in nodes], dtype=float)
#         # bc_vals = np.array([bc.get(node, 0.0) for node in nodes], dtype=float)
#         # ci_vals = np.array([ci.get(node, 1.0) for node in nodes], dtype=float)

#         # def safe_minmax(x):
#         #     xmin = x.min()
#         #     xmax = x.max()
#         #     if xmax - xmin < 1e-12:
#         #         return np.ones_like(x) * 0.5
#         #     return (x - xmin) / (xmax - xmin)

#         # tr_norm = safe_minmax(traffic_vals)  # higher better
#         # bc_norm = safe_minmax(bc_vals)       # higher better
#         # ci_norm = safe_minmax(ci_vals)       # lower better

#         # combined score: high if high traffic, high BC, low CI
#         # scores = (w_traffic * tr_norm +
#         #           w_bc * bc_norm -
#         #           w_ci * ci_norm)
#         # scores = w_ci
                  

#         # min_score = scores.min()
#         # if min_score < 0:
#         #     scores = scores - min_score + 1e-6

#         # probs = scores / (scores.sum() + 1e-12)

#         allocs = [0] * n_nodes
#         for _ in range(cache_budget):
#             idx = np.random.choice(n_nodes, p=probs)
#             allocs[idx] += 1

#     logger.info(f"In init, final allocs: {allocs}")

#     tiers_per_node = {node: copy.deepcopy(params["cache_policy"]["tiers"]) for node in nodes}

#     sol = {
#         "allocations": allocs,
#         "cache_budget": cache_budget,
#         "icr_candidates": nodes,
#         "tiers_per_node": tiers_per_node,
#         "node_carbon_intensity": ci,
#         # "node_betweenness": bc,
#         # "node_traffic": traffic,
#     }
#     return sol


# def mutate_fn(sol, w_traffic=0.6, w_bc=0.3, w_ci=0.1):
#     """
#     Traffic + BC + Carbon-aware mutation for PAES using node IDs only.
#     Preserves total cache budget; never uses positional indexes into allocs.
#     Moves cache from low-score (low traffic/BC, high CI) nodes
#     to high-score (high traffic/BC, low CI) nodes.
#     """
#     import copy
#     import random
#     import numpy as np

#     new_sol = copy.deepcopy(sol)

#     nodes = list(new_sol["icr_candidates"])   # iterable of node ids
#     alloc_list = new_sol["allocations"][:]    # list of ints
#     B = new_sol["cache_budget"]

#     # node -> allocation
#     alloc = {node: a for node, a in zip(nodes, alloc_list)}

#     traffic = new_sol.get("node_traffic", {})           # node -> traffic weight
#     bc = new_sol.get("node_betweenness", {})            # node -> BC
#     ci = new_sol.get("node_carbon_intensity", {})       # node -> CI

#     # build metric arrays aligned with nodes list
#     traffic_vals = np.array([traffic.get(n, 0.0) for n in nodes], dtype=float)
#     bc_vals = np.array([bc.get(n, 0.0) for n in nodes], dtype=float)
#     ci_vals = np.array([ci.get(n, 1.0) for n in nodes], dtype=float)

#     def safe_minmax(x):
#         xmin = x.min()
#         xmax = x.max()
#         if xmax - xmin < 1e-12:
#             return np.ones_like(x) * 0.5
#         return (x - xmin) / (xmax - xmin)

#     tr_norm = safe_minmax(traffic_vals)  # higher better
#     bc_norm = safe_minmax(bc_vals)       # higher better
#     ci_norm = safe_minmax(ci_vals)       # lower better

#     # combined score: high if high traffic, high BC, low CI
#     scores = (w_traffic * tr_norm +
#               w_bc * bc_norm -
#               w_ci * ci_norm)

#     min_score = scores.min()
#     if min_score < 0:
#         scores = scores - min_score + 1e-6

#     node_score = {n: float(s) for n, s in zip(nodes, scores)}

#     # sort nodes by score
#     sorted_nodes = sorted(nodes, key=lambda n: node_score[n])
#     n_nodes = len(sorted_nodes)
#     bottom_k = max(1, n_nodes // 3)
#     top_k = max(1, n_nodes // 3)

#     # donors: low score, alloc > 0
#     donor_candidates = [n for n in sorted_nodes[:bottom_k]
#                         if alloc.get(n, 0) > 0]
#     if not donor_candidates:
#         return new_sol
#     donor_node = random.choice(donor_candidates)

#     # receivers: high score, different node
#     receiver_candidates = [n for n in sorted_nodes[-top_k:]
#                            if n != donor_node]
#     if not receiver_candidates:
#         return new_sol
#     receiver_node = random.choice(receiver_candidates)

#     # small move (up to 5% of budget)
#     max_delta = max(1, int(B * 0.05))
#     delta = random.randint(1, max_delta)
#     delta = min(delta, alloc.get(donor_node, 0))
#     if delta == 0:
#         return new_sol

#     # apply transfer in node->alloc map
#     alloc[donor_node] = alloc.get(donor_node, 0) - delta
#     alloc[receiver_node] = alloc.get(receiver_node, 0) + delta

#     # rebuild allocation list in nodes order
#     new_allocs = [alloc.get(n, 0) for n in nodes]

#     # enforce exact budget
#     total = sum(new_allocs)
#     if total != B:
#         diff = B - total
#         if diff > 0:
#             # give extra to receiver_node
#             idx = nodes.index(receiver_node)
#             new_allocs[idx] += diff
#         else:
#             for _ in range(-diff):
#                 candidates = [i for i, a in enumerate(new_allocs) if a > 0]
#                 if not candidates:
#                     break
#                 i = random.choice(candidates)
#                 new_allocs[i] -= 1

#     new_sol["allocations"] = new_allocs
#     return new_sol

# --- Mutation function ---
# def mutate_allocations(parent_allocs, moves_per_mutation=3, rng=None):
#     """
#     Small, guaranteed-change mutation on integer cache allocations.

#     - Keeps total sum constant.
#     - Always returns a vector different from parent_allocs.
#     - Only moves 1 unit at a time => local but non-trivial exploration.
#     """
#     if rng is None:
#         rng = random

#     n = len(parent_allocs)
#     child = parent_allocs[:]

#     # Edge case: nothing to move
#     if sum(child) == 0 or n < 2:
#         return child

#     attempts = 0
#     max_attempts = 20

#     while attempts < max_attempts:
#         tmp = child[:]  # work on a copy for this attempt

#         for _ in range(moves_per_mutation):
#             # pick donor with >0
#             donor_candidates = [i for i, v in enumerate(tmp) if v > 0]
#             if not donor_candidates:
#                 break
#             d = rng.choice(donor_candidates)

#             # pick any receiver != donor
#             recv_candidates = [i for i in range(n) if i != d]
#             if not recv_candidates:
#                 break
#             r = rng.choice(recv_candidates)

#             # move 1 unit
#             tmp[d] -= 1
#             tmp[r] += 1

#         # if this attempt changed something, accept it
#         if tmp != parent_allocs:
#             return tmp

#         attempts += 1

#     # fallback: if we somehow failed, flip 1 unit deterministically
#     donor_candidates = [i for i, v in enumerate(child) if v > 0]
#     if len(donor_candidates) >= 1:
#         d = donor_candidates[0]
#         r = (d + 1) % n
#         child[d] -= 1
#         child[r] += 1

#     return child

# def mutate_fn(sol, moves_per_mutation=3, seed=None):
#     """
#     PAES-compatible mutation:
#     - only touches `allocations`
#     - guarantees a different child
#     - keeps cache_budget constant (sum of allocations)
#     """
#     rng = random.Random(seed) if seed is not None else random

#     parent_allocs = sol["allocations"]
#     child_allocs = mutate_allocations(parent_allocs,
#                                       moves_per_mutation=moves_per_mutation,
#                                       rng=rng)

#     new_sol = copy.deepcopy(sol)
#     new_sol["allocations"] = child_allocs

#     # optional sanity check: preserve budget
#     parent_sum = sum(parent_allocs)
#     child_sum = sum(child_allocs)
#     if parent_sum != child_sum:
#         # repair by simple rescaling if this ever happens
#         diff = child_sum - parent_sum
#         if diff > 0:
#             # remove diff units from random nodes with >0
#             for _ in range(diff):
#                 idxs = [i for i, v in enumerate(child_allocs) if v > 0]
#                 if not idxs:
#                     break
#                 i = rng.choice(idxs)
#                 child_allocs[i] -= 1
#         elif diff < 0:
#             # add -diff units to random nodes
#             for _ in range(-diff):
#                 i = rng.randrange(len(child_allocs))
#                 child_allocs[i] += 1
#         new_sol["allocations"] = child_allocs

#     return new_sol

def mutate_fn(sol, rng):
    logger.info("Mutate solution")
    print("Mutate solution")
    new_sol = copy.deepcopy(sol)
    allocs = new_sol["allocations"][:]
    n_nodes = len(allocs)
    
    # perturb one node allocation
    for _ in range(rng.randint(3, 8)):
        idx = rng.randrange(n_nodes)
        delta = rng.randint(-10, 10)   # larger mutation
        allocs[idx] = max(0, allocs[idx] + delta)

    total = sum(allocs)
    if total > 0:
        # scale back to original cache budget
        scaled = [a / total * sol["cache_budget"] for a in allocs]
        floored = [int(math.floor(a)) for a in scaled]

        # FIX: remainder must be an int
        remainder = int(sol["cache_budget"] - sum(floored))

        # distribute leftover to largest fractional parts
        fractions = sorted(
            enumerate([a - f for a, f in zip(scaled, floored)]),
            key=lambda x: x[1],
            reverse=True
        )
        for i, _ in fractions[:remainder]:
            floored[i] += 1
        allocs = floored

    new_sol["allocations"] = allocs
    return new_sol

# def mutate_fn(sol):
#     logger.info("Mutate solution (100% carbon-aware)")
#     print("Mutate solution (100% carbon-aware)")
#     new_sol = copy.deepcopy(sol)

#     allocs = new_sol["allocations"][:]
#     nodes = list(new_sol.get("icr_candidates", []))
#     ci = new_sol.get("node_carbon_intensity")

#     if not nodes or not ci:
#         logger.warning("No nodes or CI info in solution; mutation is no-op.")
#         return new_sol

#     # Assurer un ordre stable
#     if isinstance(nodes, set):
#         nodes = sorted(nodes)

#     # Trier les nœuds par intensité carbone (croissant)
#     rated = sorted(nodes, key=lambda x: ci.get(x, float("inf")))  # plus vert d'abord

#     k = min(3, len(rated))  # sécurité si peu de nœuds
#     green_targets = rated[:k]      # k plus verts
#     dirty_sources = rated[-k:]     # k plus sales

#     # Calculer combien on peut déplacer
#     total_to_shift = sum(allocs[nodes.index(d)] for d in dirty_sources)

#     if total_to_shift == 0:
#         logger.info("Nothing to shift from dirty nodes; mutation is no-op.")
#         return new_sol

#     # Vider les k plus sales
#     for d in dirty_sources:
#         idx = nodes.index(d)
#         allocs[idx] = 0

#     # Répartir tout vers les k plus verts
#     per_target = total_to_shift // k
#     remainder = total_to_shift % k

#     for g in green_targets:
#         idx = nodes.index(g)
#         allocs[idx] += per_target

#     # Donner le reste au plus vert
#     allocs[nodes.index(green_targets[0])] += remainder

#     new_sol["allocations"] = allocs
#     return new_sol

# def mutate_fn(sol):
#     logger.info("Mutate solution (green-aware if CI present)")
#     new_sol = copy.deepcopy(sol)
#     allocs = new_sol["allocations"][:]
#     nodes = new_sol.get("icr_candidates", [])
#     ci = new_sol.get("node_carbon_intensity")
    
#     # Convert to list if stored incorrectly
#     if isinstance(nodes, set):
#         nodes = sorted(list(nodes))

#     # Sort nodes by carbon intensity
#     rated = sorted(nodes, key=lambda x: ci[x])  # lowest first
#     green_targets = rated[:3]   # top 3 greenest nodes
#     dirty_sources = rated[-3:]  # top 3 dirtiest nodes

#     total_to_shift = sum(allocs[nodes.index(d)] for d in dirty_sources)

#     # EXTREME: shift ALL cache from 3 dirtiest → 3 greenest
#     for d in dirty_sources:
#         idx = nodes.index(d)
#         stolen = allocs[idx]
#         allocs[idx] = 0  # wipe
#         # distribute equally
#         per_target = stolen // len(green_targets)
#         for g in green_targets:
#             allocs[nodes.index(g)] += per_target

#     # If leftover from integer rounding
#     leftover = total_to_shift % len(green_targets)
#     allocs[nodes.index(green_targets[0])] += leftover  # dump remainder to greenest

#     new_sol["allocations"] = allocs

#     return new_sol

# def mutate_fn(sol, shift_fraction=0.2, k=3):
#     """
#     Incremental green-aware mutation.

#     - shift_fraction: fraction of each dirty node's cache to move (0–1)
#     - k: how many greenest / dirtiest nodes to consider
#     """
#     logger.info("Mutate solution (incremental green-aware)")

#     new_sol = copy.deepcopy(sol)
#     allocs = new_sol["allocations"][:]
#     nodes = new_sol.get("icr_candidates", [])
#     ci = new_sol.get("node_carbon_intensity")

#     # Safety checks
#     if not nodes or ci is None or len(nodes) <= 1:
#         logger.info("Mutation skipped (no nodes or CI).")
#         return new_sol

#     # Ensure deterministic ordering
#     if isinstance(nodes, set):
#         nodes = sorted(list(nodes))

#     # Rank nodes by carbon intensity (lowest = greenest)
#     rated = sorted(nodes, key=lambda x: ci[x])
#     k = min(k, len(rated) // 2 or 1)
#     green_targets = rated[:k]
#     dirty_sources = rated[-k:]

#     # Compute how much to shift from each dirty node
#     total_shift = 0
#     shifts = {}
#     for d in dirty_sources:
#         idx = nodes.index(d)
#         move = int(allocs[idx] * shift_fraction)
#         if move <= 0:
#             continue
#         shifts[d] = move
#         allocs[idx] -= move
#         total_shift += move

#     if total_shift == 0 or not green_targets:
#         logger.info("Nothing to shift; mutation is noop.")
#         new_sol["allocations"] = allocs
#         return new_sol

#     # Distribute shifted cache across green targets
#     per_target = total_shift // len(green_targets)
#     remainder = total_shift % len(green_targets)

#     for g in green_targets:
#         idx = nodes.index(g)
#         allocs[idx] += per_target

#     # Put any leftover on the greenest node
#     allocs[nodes.index(green_targets[0])] += remainder

#     new_sol["allocations"] = allocs
#     return new_sol


def compute_green_centrality_score(nodes, ci, centrality,
                                   w_central=0.5, w_carbon=0.5):
    # Normalize centrality and carbon
    c_vals = [centrality.get(n, 0.0) for n in nodes]
    ci_vals = [ci.get(n, 1.0) for n in nodes]

    c_min, c_max = min(c_vals), max(c_vals)
    ci_min, ci_max = min(ci_vals), max(ci_vals)

    scores = {}
    for n in nodes:
        c = centrality.get(n, 0.0)
        ci_n = ci.get(n, 1.0)

        # centrality: higher is better
        if c_max > c_min:
            c_norm = (c - c_min) / (c_max - c_min)
        else:
            c_norm = 0.0

        # carbon: lower is better → invert
        if ci_max > ci_min:
            ci_norm = (ci_n - ci_min) / (ci_max - ci_min)
        else:
            ci_norm = 0.0
        green_norm = 1.0 - ci_norm

        scores[n] = w_central * c_norm + w_carbon * green_norm

    return scores


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
    exp["workload"]["n_measured"] /= 2
     # tiers logic unchanged
    if tiers_per_node is not None:
        exp["cache_policy"]["tiers_per_node"] = tiers_per_node
    
    return exp


def eval_fn(sol, **kwargs):
    params  = kwargs["params"]
    metrics = kwargs["metrics"]
    settings = kwargs["settings"]
    
    results_file = EXAMPLES_DIR / "paes_results.pickle"
    config_file = EXAMPLES_DIR / "paes_config.py"
    exp_pkl = EXAMPLES_DIR / "paes.pkl"
    
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
        f.write(f"PARALLEL_EXECUTION = {settings.PARALLEL_EXECUTION}\n")
        f.write(f"N_REPLICATIONS = {settings.N_REPLICATIONS}\n")
        f.write(f"N_PERIODS = 1\n")
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
        return (None, None, None)


def run_paes(icr_candidates, params, metrics, settings, cache_budget, archive_size, grid_divisions, max_evaluations, seed, allocs):
    logger.info("Run PAES")
        # Create a local RNG for this PAES instance
    

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

if __name__ == "__main__":
    # Find Pareto solutions
    ICR_CANDIDATES = [
        4, 6, 10, 14, 15, 17, 18, 20, 21, 22,
        29, 31, 34, 35, 36, 37, 38, 39, 40, 44,
        45, 46, 49, 55, 56, 58, 59
    ]
    pareto = run_paes(icr_candidates=ICR_CANDIDATES,
                    topology="GARR",
                    alpha=1.2,
                    strategy="CL2SM",
                    network_cache=0.015,
                    cache_placement="UNIFORM",
                    archive_size=40,
                    grid_divisions=30,
                    max_evaluations=2,  # small for test, increase later
                    seed=0)
    
    print("Found", len(pareto), "Pareto solutions (max Hit, min Cost, min Carbon):")
    for sol, (h, c, cf) in pareto:
        print(sol["allocations"], " -> hit=%.6g, cost=%.6f, carbon=%.6f" % (-h, c, cf))
    
    # Combined plot
    hits = [-obj[0] for _, obj in pareto]  # invert -hit for plot
    costs = [obj[1] for _, obj in pareto]
    carbons = [obj[2] for _, obj in pareto]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(hits, costs, carbons, c="royalblue", s=60, label="PAES Pareto")

    ax.set_xlabel("Hit Rate (maximize)")
    ax.set_ylabel("Cost (minimize)")
    ax.set_zlabel("Carbon (minimize)")
    ax.set_title("PAES Pareto Front vs Baselines")
    ax.legend()
    plt.tight_layout()

    out_path = EXAMPLES_DIR / "paes_logs/paes_vs_baselines_3d.png"
    fig.savefig(out_path, dpi=300)
    plt.close()
    print(f"\n📌 3D plot saved to: {out_path}")
