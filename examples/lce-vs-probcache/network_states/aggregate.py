#!/usr/bin/env python3

import glob
import json
import pickle
import re
from copy import deepcopy

from config import EXPERIMENT_QUEUE
from icarus.util import Tree
from icarus.results import ResultSet

# -------------------------------------------------------------------
# Paths / patterns
# -------------------------------------------------------------------

# JSON_PATTERN = "examples/lce-vs-probcache/network_states/exp*_TELEKOM_p*.json"
JSON_PATTERN = "examples/lce-vs-probcache/network_states/exp*_GEANT_p*.json"
# JSON_PATTERN = "examples/lce-vs-probcache/network_states/exp*_GARR_p*.json"
FNAME_RE = re.compile(r"exp(?P<exp>\d+)_GEANT_p(?P<period>\d+)\.json")
# FNAME_RE = re.compile(r"exp(?P<exp>\d+)_GARR_p(?P<period>\d+)\.json")
# FNAME_RE = re.compile(r"exp(?P<exp>\d+)_TELEKOM_p(?P<period>\d+)\.json")

ORIG_RESULTS_PKL = "examples/lce-vs-probcache/results.pickle"
# ORIG_RESULTS_PKL = "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results_aggregated.pickle"
OUT_RESULTS_PKL  = "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results_aggregated.pickle"


# -------------------------------------------------------------------
# Discover JSON files
# -------------------------------------------------------------------

def discover_files_by_experiment():
    files = sorted(glob.glob(JSON_PATTERN))
    by_exp = {}
    for path in files:
        fname = path.split("/")[-1]
        m = FNAME_RE.match(fname)
        if not m:
            continue
        exp_id = int(m.group("exp"))
        period = int(m.group("period"))
        by_exp.setdefault(exp_id, []).append((period, path))

    for exp_id in by_exp:
        by_exp[exp_id].sort(key=lambda x: x[0])

    return by_exp


# -------------------------------------------------------------------
# Load JSON old_results
# -------------------------------------------------------------------

def load_period_old_results(period_paths):
    out = []
    for _, path in period_paths:
        with open(path) as f:
            data = json.load(f)
        out.append(data["old_results"])
    return out


# -------------------------------------------------------------------
# Aggregate metrics
# -------------------------------------------------------------------

def aggregate_old_results(period_results):
    n = len(period_results)
    base = deepcopy(period_results[0])

    # Average MEAN fields
    for k, v in base.items():
        if isinstance(v, dict) and "MEAN" in v:
            base[k]["MEAN"] = sum(
                pr[k]["MEAN"] for pr in period_results
            ) / n

    # Sum carbon footprint components
    if "CARBONFOOTPRINT" in base:
        for subk, v in base["CARBONFOOTPRINT"].items():
            if isinstance(v, (int, float)):
                base["CARBONFOOTPRINT"][subk] = sum(
                    pr["CARBONFOOTPRINT"][subk]
                    for pr in period_results
                )

    return base


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    # ---- Load original results ----
    with open(ORIG_RESULTS_PKL, "rb") as f:
        orig_rs: ResultSet = pickle.load(f)

    # Index by cache placement NAME (THIS IS THE FIX)
    placement_to_entry = {}
    for cfg, res in orig_rs:
        try:
            name = cfg["cache_placement"]["name"]
            network_cache = cfg["cache_placement"]["network_cache"]
            alpha = cfg["workload"]["alpha"]
        except KeyError as e:
            raise KeyError(f"Missing key in baseline config: {e}")
        key = (name, network_cache, alpha)
        if key in placement_to_entry:
            raise RuntimeError(
                f"Duplicate baseline detected for {key}. "
                f"Baselines must be unique per (placement, network_cache)."
            )
        placement_to_entry[key] = (cfg, res)
        res_new = res
    # Fill missing entries from EXPERIMENT_QUEUE
    for experiment in EXPERIMENT_QUEUE:
        name = experiment["cache_placement"]["name"]
        network_cache = experiment["cache_placement"]["network_cache"]
        alpha = experiment["workload"]["alpha"]
        key = (name, network_cache, alpha)

        if key not in placement_to_entry:
            cfg_new = experiment
            orig_rs.add(cfg_new, res_new)
            placement_to_entry[key] = (cfg_new, res_new)
            print(f"Created missing baseline for {key}")
    by_exp_files = discover_files_by_experiment()
    new_rs = ResultSet()

    # ---- Aggregate per experiment ----
    for exp_id, period_paths in sorted(by_exp_files.items()):
        _, first_path = period_paths[0]
        with open(first_path) as f:
            data0 = json.load(f)

        try:
            placement_name = data0["cache_placement"]
            network_cache = data0["network_cache"]
            alpha = data0["alpha"]
        except KeyError as e:
            raise KeyError(
                f"Missing key in period JSON (exp {exp_id}): {e}"
            )
        key = (placement_name, network_cache, alpha)

        if key not in placement_to_entry:
            print(f"Missing baseline for {key}")
            continue

        cfg_orig, _ = placement_to_entry[key]

        period_results = load_period_old_results(period_paths)
        aggregated = aggregate_old_results(period_results)

        # ---- Create independent result ----
        cfg_new = deepcopy(cfg_orig)
        cfg_new["experiment_id"] = exp_id

        res_new = Tree({})   # start clean
        for k, v in aggregated.items():
            res_new[k] = Tree(v)

        new_rs.add(cfg_new, res_new)

        print(f"Aggregated exp{exp_id} → {placement_name}")

    # ---- Save ----
    with open(OUT_RESULTS_PKL, "wb") as f:
        pickle.dump(new_rs, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved:", OUT_RESULTS_PKL)


if __name__ == "__main__":
    main()
