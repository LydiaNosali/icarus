import pickle
from pathlib import Path
import copy
import random
import math

from matplotlib import pyplot as plt
from paes import PAES
from icarus.runner import run
from config import default, strategy_params, TIERS, DATA_COLLECTORS

# Path to the examples directory where Makefile and results are generated
EXAMPLES_DIR = Path(__file__).parent
ICR_CANDIDATES = [
    4, 6, 10, 14, 15, 17, 18, 20, 21, 22,
    29, 31, 34, 35, 36, 37, 38, 39, 40, 44,
    45, 46, 49, 55, 56, 58, 59
]
TOPOLOGY = "GARR"
ALPHA = 1.2
NETWORK_CACHE = 0.015
STRATEGY = "CL2SM"
BASELINES = ["UNIFORM", "RANDOM", "BETWEENNESS_CENTRALITY"]
n_contents = default["workload"]["n_contents"]
cache_budget = int(round(n_contents * NETWORK_CACHE))
    
def init_fn():
    results_file = EXAMPLES_DIR / "results.pickle"
    config_file = EXAMPLES_DIR / "paes_config.py"
    exp_pkl = EXAMPLES_DIR / "exp.pkl"
    
    # --- Step 1: Build an experiment with init placement ---
    exp = build_experiment(
        topology=TOPOLOGY,
        alpha=ALPHA,
        strategy=STRATEGY,
        network_cache=NETWORK_CACHE,
        allocations=[],
        tiers_per_node=None,
        cache_placement="BETWEENNESS_CENTRALITY"
    )

    with open(exp_pkl, "wb") as f:
        pickle.dump([exp], f)

    with open(config_file, "w") as f:
        f.write("from collections import deque\n")
        f.write("from config import *\n")
        f.write("import pickle\n")
        f.write("EXPERIMENT_QUEUE = deque()\n")
        f.write("EXPERIMENT_QUEUE.extend(pickle.load(open('exp.pkl','rb')))\n")

    # --- Step 2: Run Icarus once so cache_allocations get filled ---
    run(str(config_file), str(results_file), {})

    # --- Step 3: Read back the allocations from results.pickle ---
    with open(results_file, "rb") as f:
        results = pickle.load(f)
    _, metrics = results[0]  # params, metrics

    alloc_dict = metrics.get("cache_allocations", {})
    allocs = [alloc_dict.get(v, 0) for v in ICR_CANDIDATES]
    tiers_per_node = {node: copy.deepcopy(TIERS) for node in ICR_CANDIDATES}

    return {
        "allocations": allocs,
        "alpha": 1.2,
        "topology": TOPOLOGY,
        "network_cache": NETWORK_CACHE,
        "tiers_per_node": tiers_per_node,   # 👈 store per-node
    }

# --- Mutation function ---
def mutate_fn(sol):
    new_sol = copy.deepcopy(sol)
    allocs = new_sol["allocations"][:]
    n_nodes = len(allocs)
    
    # perturb one node allocation
    for _ in range(random.randint(3, 8)):
        idx = random.randrange(n_nodes)
        delta = random.randint(-30, 30)   # larger mutation
        allocs[idx] = max(0, allocs[idx] + delta)

    total = sum(allocs)
    if total > 0:
        scaled = [a / total * cache_budget for a in allocs]
        floored = [int(math.floor(a)) for a in scaled]
        remainder = cache_budget - sum(floored)
        # distribute leftover to largest fractional parts
        fractions = sorted(
            enumerate([a - f for a, f in zip(scaled, floored)]),
            key=lambda x: x[1], reverse=True
        )
        for i, _ in fractions[:remainder]:
            floored[i] += 1
        allocs = floored
    new_sol["allocations"] = allocs
    
    # --- Mutate tiers (per node) ---
    node = random.choice(list(new_sol["tiers_per_node"].keys()))
    tiers = new_sol["tiers_per_node"][node]
    dram_share = random.uniform(0.05, 0.8)
    ssd_share = 1 - dram_share
    for t in tiers:
        if t["name"] == "DRAM":
            t["size_factor"] = dram_share
        elif t["name"] == "SSD":
            t["size_factor"] = ssd_share
    new_sol["tiers_per_node"][node] = tiers
    return new_sol

# --- Build experiment with explicit args ---
def build_experiment(topology, alpha, strategy, network_cache, allocations, tiers_per_node, cache_placement="ALLOCATED"):
    exp = copy.deepcopy(default)
    exp["topology"]["name"] = topology
    exp["strategy"]["name"] = strategy
    if strategy in strategy_params:
        exp["strategy"].update(strategy_params[strategy])
    exp["workload"]["alpha"] = alpha
    exp["data_collectors"] = copy.deepcopy(DATA_COLLECTORS)

    exp["cache_placement"]["name"] = cache_placement
    exp["cache_placement"]["network_cache"] = network_cache

    # Placeholder mapping: node indices to allocations
    if cache_placement == "ALLOCATED":
        exp["cache_placement"]["allocations"] = {
            node: int(a) for node, a in zip(ICR_CANDIDATES, allocations)
        }
    
    # Attach tier configuration
    if tiers_per_node is not None:
        exp["cache_policy"]["tiers_per_node"] = tiers_per_node
    elif TIERS is not None:
        base_tiers_per_node = {node: copy.deepcopy(TIERS) for node in ICR_CANDIDATES}
        exp["cache_policy"]["tiers_per_node"] = base_tiers_per_node
    return exp

# --- Evaluation function ---
def eval_fn(sol):
    exp = build_experiment(
        topology=sol["topology"],
        alpha=sol["alpha"],
        strategy="CL2SM",
        network_cache=sol["network_cache"],
        allocations=sol["allocations"],
        tiers_per_node=sol["tiers_per_node"],
        cache_placement="ALLOCATED"
    )
    # --- Patch collectors to use per-node tiers ---
    if "tiers" in sol:
        exp["data_collectors"]["COST"]["tiers"] = exp["cache_placement"]["tiers_per_node"]
        exp["data_collectors"]["CARBONFOOTPRINT"]["tiers"] = exp["cache_placement"]["tiers_per_node"]

    print(f"Evaluating allocations:{[a for a in sol["allocations"]]}, sum : {sum(sol["allocations"])}")
    # print(f"Evaluating tiers:{[a for a in sol["tiers_per_node"]]}")
    
    exp_pkl = EXAMPLES_DIR / "exp.pkl"
    with open(exp_pkl, "wb") as f:
        pickle.dump([exp], f)

    config_file = EXAMPLES_DIR / "paes_config.py"
    with open(config_file, "w") as f:
        f.write("from collections import deque\n")
        f.write("from config import *\n")
        f.write("import pickle\n")
        f.write("EXPERIMENT_QUEUE = deque()\n")
        f.write("EXPERIMENT_QUEUE.extend(pickle.load(open('exp.pkl','rb')))\n")

    results_file = EXAMPLES_DIR / "results.pickle"

    try:
        run(str(config_file), str(results_file), {})
    except Exception as e:
        print("Icarus failed:", e)
        return (1e12, 1e12)

    if not results_file.exists():
        return (1e12, 1e12)

    try:
        with open(results_file, "rb") as f:
            data = pickle.load(f)
        
        _, metrics = data._results[0]
        cost = metrics.get("COST").get("MEAN")
        carbon = metrics.get("CARBONFOOTPRINT").get("MEAN")
        print(f"cost : {cost}, carbon : {carbon}")
        return (cost, carbon)

    except Exception as e:
        print("Error reading results:", e)
        return (1e12, 1e12)

# --- Run PAES ---
if __name__ == "__main__":
    opt = PAES(
        init_fn=init_fn,
        mutate_fn=mutate_fn,
        eval_fn=eval_fn,
        archive_size=40,
        grid_divisions=30,
        max_evaluations=10,  # small for test, increase later
        seed=0,
    )
    # Find Pareto solutions
    pareto = opt.run()
    print("Found", len(pareto), "Pareto solutions (min Cost, min Carbon):")
    for sol, (c, cf) in pareto:
        print(sol, " ->  cost=%.6f, carbon=%.6f" % (c, cf))
    
    # Evaluate baselines (ICARUS)
    def evaluate_baseline(cache_placement):
        tiers_per_node = {node: copy.deepcopy(TIERS) for node in ICR_CANDIDATES}
        exp = build_experiment(
            topology=TOPOLOGY,
            alpha=ALPHA,
            strategy="CL2SM",
            network_cache=NETWORK_CACHE,
            allocations=[],
            tiers_per_node=tiers_per_node,
            cache_placement=cache_placement,
        )
        exp_pkl = EXAMPLES_DIR / "exp.pkl"
        with open(exp_pkl, "wb") as f:
            pickle.dump([exp], f)
        return eval_fn({
            "topology": TOPOLOGY,
            "alpha": ALPHA,
            "network_cache": NETWORK_CACHE,
            "allocations": [],
            "tiers_per_node": tiers_per_node,
        })

    baseline_results = {}
    for placement in BASELINES:
        print(f"\nEvaluating baseline: {placement}")
        cost, carbon = evaluate_baseline(placement)
        baseline_results[placement] = (cost, carbon)
        print(f"  → cost={cost:.6f}, carbon={carbon:.6e}")

    # Combined plot
    costs = [obj[0] for _, obj in pareto]
    carbons = [obj[1] for _, obj in pareto]

    plt.figure(figsize=(8, 6))
    plt.scatter(costs, carbons, c="royalblue", s=60, label="PAES Pareto Front")

    markers = {
        "UNIFORM": "P",
        "RANDOM": "X",
        "BETWEENNESS_CENTRALITY": "*",
    }
    for name, (cost, carbon) in baseline_results.items():
        plt.scatter(cost, carbon, s=180, marker=markers.get(name, "o"), label=name.title())
        plt.text(cost * 1.0005, carbon * 1.05, name.title(), fontsize=10, weight="bold")

    plt.xlabel("Cost", fontsize=12)
    plt.ylabel("Carbon Footprint", fontsize=12)
    plt.title("PAES Pareto Front vs Icarus Baselines", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_path = Path("paes_logs/paes_vs_baselines.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\n✅ Saved combined plot to {out_path}")

    # Dominance check
    def dominates(a, b):
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

    for name, (cost_b, carbon_b) in baseline_results.items():
        dominated = any(dominates(obj, (cost_b, carbon_b)) for _, obj in pareto)
        status = "dominated by PAES" if dominated else "non-dominated"
        print(f"Baseline {name}: {status}")

