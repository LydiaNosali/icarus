import json
import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# with open("examples/lce-vs-probcache/network_states/exp1_GARR.json") as f:
#     state = json.load(f)

# cache_size_diff = {}
# for node_id, cache_size in state["cache_size"].items():
#     cache_size_diff[node_id] = cache_size - state["old_cache_size"].get(node_id)

# for node_id, node_data in state["old_per_node_tiers"].items():
#     if cache_size_diff[node_id] < 0:
#         print("/////")
#         print(node_id)
#         tiers = node_data["tiers"]
#         tiers_sorted = list(tiers.keys())
#         for i, tier_name in enumerate(tiers_sorted):
#             current_max_len = state["per_node_tiers"].get(node_id)["tiers"][tier_name]["maxlen"]
#             tmaxlen = tiers[tier_name]["maxlen"]
#             if current_max_len < tmaxlen:
#                 print(state["per_node_tiers"].get(node_id)["tiers"][tier_name])

#///////////////////////////////////////////////////////////////////////////////////
# results_file = "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results.pickle"
# F1_hit = []
# F2_carbon = []
# F3_latency = []
# F4_cost = []
# with open(results_file, "rb") as f:
#         data = pickle.load(f)
#         _, metrics = data._results[0]
#         for _,metrics in data._results:
                # print(metrics)
                # cost = metrics.get("COST").get("MEAN")
                # carbon = metrics.get("CARBONFOOTPRINT").get("TOTAL")
                # hit = metrics.get("CACHE_HIT_RATIO").get("MEAN")
                # latency = metrics.get("LATENCY").get("MEAN")
                
                # Append to lists
                # F1_hit.append(hit)
                # F2_carbon.append(carbon)
                # F3_latency.append(latency)
                # F4_cost.append(cost)

# Convert to numpy arrays
# F1 = np.array(F1_hit)
# F2 = np.array(F2_carbon)
# F3 = np.array(F3_latency)
# F4 = np.array(F4_cost)

# Optional: print arrays to check
# print("F1 (Hit rates):", F1)
# print("F2 (Carbon):", F2)
# print("F3 (Latency):", F3)
# print("F4 (Cost):", F4)

# Stack objectives into a single matrix (rows = objectives, columns = samples)
# objectives = np.vstack([F1, F2, F3, F4])

# Pearson correlation
# pearson_corr = np.corrcoef(objectives)
# print("Pearson correlation matrix:")
# print(pearson_corr)

# Spearman correlation
# spearman_corr, _ = spearmanr(objectives.T)  # transpose: rows=samples, cols=objectives
# print("\nSpearman correlation matrix:")
# print(spearman_corr)

# Heatmap for Pearson correlation
# plt.figure(figsize=(6,5))
# sns.heatmap(pearson_corr, annot=True, xticklabels=['Hit','Carbon','Latency','Cost'],
#             yticklabels=['Hit','Carbon','Latency','Cost'], cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 14})
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.title("Pearson Correlation of Objectives", fontsize=20)
# plt.tight_layout()
# plt.savefig("pearson_corr_heatmap2.png", dpi=300, bbox_inches="tight")
# plt.show()

# Heatmap for Spearman correlation
# plt.figure(figsize=(6,5))
# sns.heatmap(spearman_corr, annot=True, xticklabels=['Hit','Carbon','Latency','Cost'],
#             yticklabels=['Hit','Carbon','Latency','Cost'], cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 14})
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.title("Spearman Correlation Coefficients", fontsize=20)
# plt.tight_layout()
# plt.savefig("/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/correlation/spearman_corr_heatmap.png", dpi=300, bbox_inches="tight")
# # plt.show()
# def classify_corr(corr):
#     if corr < -0.2:
#         return "negative"
#     elif corr > 0.2:
#         return "positive"
#     else:
#         return "insignificant"

# Corrélations Pearson
# corr_12 = np.corrcoef(F1, F2)[0,1]
# corr_14 = np.corrcoef(F1, F4)[0,1]
# corr_24 = np.corrcoef(F2, F4)[0,1]

# pairs = [("Hit, Cost", corr_14), ("Hit, Carbon", corr_12), ("Carbon, Cost", corr_24)]
# categories = ["negative", "insignificant", "positive"]

# Calcul des pourcentages (cas unique => 100% dans la catégorie mesurée)
# results = {p: classify_corr(c) for p,c in pairs}

# freqs = {p: {cat: (1 if results[p]==cat else 0) * 100 for cat in categories} for p,_ in pairs}

# === Graphique en barres ===
# labels = ["(F1,F4)", "(F1,F2)", "(F2,F4)"]
# neg = [freqs[p]["negative"] for p,_ in pairs]
# ins = [freqs[p]["insignificant"] for p,_ in pairs]
# pos = [freqs[p]["positive"] for p,_ in pairs]

# x = np.arange(len(labels))
# width = 0.6

# plt.figure(figsize=(8,5))
# plt.bar(x, neg, width, label='Negative correlation', color='blue')
# plt.bar(x, ins, width, bottom=neg, label='Insignificant correlation', color='lightgray')
# plt.bar(x, pos, width, bottom=np.array(neg)+np.array(ins), label='Positive correlation', color='purple')

# plt.xticks(x, labels, fontsize=12)
# plt.yticks(fontsize=12)
# plt.ylim(0,100)
# plt.ylabel("Percentage", fontsize=14)
# plt.title("Correlation Types Between Objective Pairs", fontsize=20)
# plt.legend(fontsize=16)
# plt.tight_layout()
# plt.savefig("correlation_types_bar.png", dpi=300, bbox_inches="tight")

#///////////////////////////////////////////////////////////////////////

# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt

# # === Your data ===
# F1 = np.array([0.698988,0.757714,0.699238,0.188666,0.188246,0.213394,0.213394,0.232566,
# 0.232566,0.234542,0.73496,0.991698,0.991696,0.996186,0.264218,0.99462,
# 0.99462,0.997498,0.995994,0.286118,0.73496,0.75681,0.786292,0.995994,
# 0.998116,0.75681,0.804756])

# F2 = np.array([3.34347304e-04,3.30600044e-04,3.34582144e-04,7.80196914e-04,
# 7.79340281e-04,7.57941590e-04,7.58775292e-04,7.68724607e-04,
# 7.71776978e-04,9.29082571e-04,3.04900692e-04,2.04873305e-05,
# 2.04814348e-05,2.32747159e-05,9.24588746e-04,1.88499777e-05,
# 1.85309481e-05,3.75391433e-05,2.13421531e-05,1.02694429e-03,
# 3.06166833e-04,2.92848160e-04,3.31327929e-04,1.71733847e-05,
# 4.08607798e-05,2.89890428e-04,3.43716543e-04])

# F4 = np.array([0.10679541,0.09865159,0.1067854,0.17623642,0.17626413,0.17290273,
# 0.17290273,0.17056399,0.17056399,0.17091881,0.10052709,0.06414177,
# 0.06414238,0.06339512,0.16725323,0.06327914,0.06327914,0.06292013,
# 0.06288649,0.16452948,0.10052709,0.096589,0.09327907,0.06288649,
# 0.06261247,0.096589,0.08984714])

# # === Parameters ===
# WINDOW = 5
# THRESHOLD = 0.2  # insignificant threshold

# # === Function to classify correlation ===
# def classify_corr(c):
#     if c < -THRESHOLD: return "negative"
#     elif c > THRESHOLD: return "positive"
#     return "insignificant"

# # === Window-based correlation classification ===
# def window_corr(a, b):
#     categories = {"negative":0, "insignificant":0, "positive":0}
#     for i in range(len(a)-WINDOW+1):
#         corr = np.corrcoef(a[i:i+WINDOW], b[i:i+WINDOW])[0,1]
#         categories[classify_corr(corr)] += 1
#     total = sum(categories.values())
#     for k in categories: categories[k] = categories[k]*100/total
#     return categories

# # === Compute results for each pair ===
# r_14 = window_corr(F1, F4)
# r_12 = window_corr(F1, F2)
# r_24 = window_corr(F2, F4)

# results = {"(F1,F4)": r_14, "(F1,F2)": r_12, "(F2,F4)": r_24}

# # === Plot ===
# labels = list(results.keys())
# neg = [results[k]["negative"] for k in labels]
# ins = [results[k]["insignificant"] for k in labels]
# pos = [results[k]["positive"] for k in labels]

# x = np.arange(len(labels))
# plt.figure(figsize=(8,5))
# plt.bar(x, neg, 0.6, color='blue', label="Negative")
# plt.bar(x, ins, 0.6, bottom=neg, color='lightgray', label="Insignificant")
# plt.bar(x, pos, 0.6, bottom=np.array(neg)+np.array(ins), color='purple', label="Positive")

# plt.ylabel("Percentage")
# plt.title("Correlation Types Between Objective Pairs (Sliding Window)")
# plt.xticks(x, labels)
# plt.ylim(0,100)
# plt.legend()
# plt.savefig("objective_corr_window.png", dpi=300)
# plt.savefig("objective_corr_window.pdf", dpi=300)
# plt.show()

# print("\n📌 Window-based results:")
# for k,v in results.items():
#     print(k, v)


# def save_state(self, filename_prefix="network_state", directory="network_states"):
    #     Path(directory).mkdir(exist_ok=True)
    #     filepath = Path(directory) / f"{filename_prefix}.pkl"

    #     # ===============================================================
    #     # Helper: evict a key consistently from a specific tier in node_state
    #     # ===============================================================
    #     def _evict_from_tier(node_state, tinfo, qname):
    #         q = tinfo[qname]
    #         if not q:
    #             return None

    #         removed = q.pop()  # LRU (right side)

    #         # Remove from global ARC queues
    #         for gq in ["t1", "t2", "b1", "b2"]:
    #             try:
    #                 node_state["global"][gq].remove(removed)
    #             except ValueError:
    #                 pass

    #         # Remove from global cache
    #         node_state["global"]["_cache"].pop(str(removed), None)

    #         tinfo[qname] = q
    #         return removed

    #     # ===============================================================
    #     # 1) Evolve carbon intensities (±5%)
    #     # ===============================================================
    #     old_ci = copy.deepcopy(self.node_carbon_intensity)
    #     new_ci = {}
    #     for node, val in old_ci.items():
    #         delta = random.uniform(-0.05, 0.05)
    #         new_val = max(0.05, min(1.0, val + delta))
    #         new_ci[node] = round(new_val, 3)

    #     # ===============================================================
    #     # 2) Compute embodied carbon per node
    #     # ===============================================================
    #     per_node_embodied = {}
    #     for node in self.cache_size:
    #         total_emb = 0.0
    #         for tier in self.per_node_tiers.get(node, []):
    #             embodied_per_gb = tier["embodied_kgco2e_per_gb"] / 1024**3
    #             total_emb += embodied_per_gb * tier["actual_size_bytes"]
    #         per_node_embodied[node] = total_emb

    #     # ===============================================================
    #     # 3) Retrieve per-node cache hit ratio from collector
    #     # ===============================================================
    #     per_node_hit_ratio = {}
    #     collector_proxy = getattr(self, "collector_proxy", None)
    #     last_period_index = getattr(self, "period", None)

    #     if collector_proxy is not None:
    #         try:
    #             if last_period_index > 0:
    #                 root = collector_proxy.results()
    #                 per_node = root.get("CACHE_HIT_RATIO").get("PER_NODE_CACHE_HIT_RATIO")
    #                 for node, chr in per_node.items():
    #                     per_node_hit_ratio[node] = chr
    #         except Exception as e:
    #             print(f"[⚠️] Could not read cache hit ratios: {e}")

    #     # ===============================================================
    #     # 4) Normalize and compute multi-criteria scores
    #     # ===============================================================
    #     def _normalize(metric_dict, invert=False):
    #         if not metric_dict:
    #             return {}
    #         vals = list(map(float, metric_dict.values()))
    #         vmin, vmax = min(vals), max(vals)
    #         if vmax == vmin:
    #             return {k: 1.0 for k in metric_dict}
    #         out = {}
    #         for k, v in metric_dict.items():
    #             x = (float(v) - vmin) / (vmax - vmin)
    #             if invert:
    #                 x = 1.0 - x
    #             out[k] = max(0.0, min(1.0, x))
    #         return out

    #     norm_hit      = _normalize(per_node_hit_ratio, invert=False)
    #     norm_ci_green = _normalize(new_ci, invert=True)
    #     norm_emb_green= _normalize(per_node_embodied, invert=True)

    #     w_hit = 0.5
    #     w_ci  = 0.25
    #     w_emb = 0.25

    #     def score(node):
    #         return (
    #             w_hit * norm_hit.get(node, 0.0)
    #             + w_ci  * norm_ci_green.get(node, 0.0)
    #             + w_emb * norm_emb_green.get(node, 0.0)
    #         )

    #     raw_scores = {node: score(node) for node in self.cache.keys()}

    #     if not raw_scores or all(v <= 0 for v in raw_scores.values()):
    #         if raw_scores:
    #             equal = 1.0 / len(raw_scores)
    #             normalized_scores = {n: equal for n in raw_scores}
    #         else:
    #             normalized_scores = {}
    #     else:
    #         total_score = sum(raw_scores.values())
    #         normalized_scores = {n: v / total_score for n, v in raw_scores.items()}

    #     # ===============================================================
    #     # 5) Compute new cache sizes (constrained to total budget)
    #     # ===============================================================
    #     total_cache_budget = sum(self.cache_size.values())
    #     new_cache_sizes = {}
    #     for node in self.cache_size:
    #         sc = normalized_scores.get(node, 0.0)
    #         alloc = max(1, round(total_cache_budget * sc)) if sc > 0 else 1
    #         new_cache_sizes[node] = alloc

    #     # ===============================================================
    #     # 6) Rebuild tier structures according to new cache sizes
    #     # ===============================================================
    #     old_cache_sizes = copy.deepcopy(self.cache_size)

    #     old_per_node_tiers = {}
    #     new_per_node_tiers = {}

    #     for node, cache_obj in self.cache.items():
    #         try:
    #             node_state = cache_obj.node_state()
    #             old_per_node_tiers[node] = copy.deepcopy(node_state)

    #             total_new = new_cache_sizes[node]

    #             # Sort tiers fastest → slowest (by latency)
    #             tier_configs = sorted(
    #                 self.per_node_tiers[node],
    #                 key=lambda t: float(t.get("latency", 1.0)),
    #             )
    #             # Keep order from tier_configs to define cascade order
    #             tiers_sorted = [
    #                 t["name"] for t in tier_configs
    #                 if t["name"] in node_state["tiers"]
    #             ]

    #             # -------------------------------------------
    #             # Compute new maxlen per tier from size_factor
    #             # -------------------------------------------
    #             new_maxlens = {}
    #             for tier in tier_configs:
    #                 name = tier["name"]
    #                 if name not in node_state["tiers"]:
    #                     continue
    #                 new_len = max(1, round(tier["size_factor"] * total_new))
    #                 new_maxlens[name] = new_len

    #             # apply to tiers
    #             for name, length in new_maxlens.items():
    #                 node_state["tiers"][name]["maxlen"] = length
                
    #             # -------------------------------------------
    #             # Build current sizes and capacity targets
    #             # -------------------------------------------
    #             curr = {}
    #             tgt = {}
    #             for name in tiers_sorted:
    #                 tinfo = node_state["tiers"][name]
    #                 curr[name] = len(tinfo["t1"]) + len(tinfo["t2"])
    #                 tgt[name] = new_maxlens[name]

    #             # ======================================================
    #             # Phase 1: STRICT CASCADE fast → slow
    #             #          DRAM surplus → SSD, SSD surplus → HDD, ...
    #             # ======================================================
    #             for i, fast_name in enumerate(tiers_sorted[:-1]):
    #                 t_fast = node_state["tiers"][fast_name]

    #                 # While this fast tier has surplus, push LRU to next slower tier
    #                 while curr[fast_name] > tgt[fast_name] and (t_fast["t1"] or t_fast["t2"]):
    #                     # 1) take one LRU item from fast tier (t2 first, then t1)
    #                     src_qname = "t2" if t_fast["t2"] else "t1"
    #                     k = t_fast[src_qname].pop()  # LRU
    #                     curr[fast_name] -= 1

    #                     # 2) move it to the next slower tier (immediate successor)
    #                     slow_name = tiers_sorted[i + 1]
    #                     t_slow = node_state["tiers"][slow_name]

    #                     # keep same ARC queue type (Option A)
    #                     dst_qname = src_qname
    #                     t_slow[dst_qname].insert(0, k)  # MRU in slow tier
    #                     curr[slow_name] += 1

    #                     # note: we do NOT touch node_state["global"] lists here.
    #                     # When/if this key is finally evicted from the slowest tier,
    #                     # _evict_from_tier() will remove it from global structures.

    #             # ======================================================
    #             # Phase 2: Evict ONLY from the slowest tier (final sink)
    #             # ======================================================
    #             if tiers_sorted:
    #                 slowest = tiers_sorted[-1]
    #                 tinfo = node_state["tiers"][slowest]
    #                 over_last = max(0, curr[slowest] - tgt[slowest])

    #                 while over_last > 0 and (tinfo["t2"] or tinfo["t1"]):
    #                     if tinfo["t2"]:
    #                         _evict_from_tier(node_state, tinfo, "t2")
    #                     else:
    #                         _evict_from_tier(node_state, tinfo, "t1")
    #                     curr[slowest] -= 1
    #                     over_last -= 1

    #             # ======================================================
    #             # Phase 3: Safety guard (per-tier enforcement)
    #             #          (should be mostly no-op; just in case of rounding)
    #             # ======================================================
    #             for name in tiers_sorted:
    #                 tinfo = node_state["tiers"][name]
    #                 maxlen = tgt[name]
    #                 size_now = len(tinfo["t1"]) + len(tinfo["t2"])
    #                 extra = max(0, size_now - maxlen)

    #                 while extra > 0 and (tinfo["t2"] or tinfo["t1"]):
    #                     if tinfo["t2"]:
    #                         _evict_from_tier(node_state, tinfo, "t2")
    #                     else:
    #                         _evict_from_tier(node_state, tinfo, "t1")
    #                     curr[name] -= 1
    #                     extra -= 1

    #             new_per_node_tiers[node] = node_state

    #         except Exception as e:
    #             print(f"[⚠️] Failed to update tiers for node {node}: {e}")

    #     # ===============================================================
    #     # 7) Rebuild global tier statistics
    #     # ===============================================================
    #     new_tier_stats = {}
    #     new_tier_sizes_mb = {}
    #     for node, node_state in new_per_node_tiers.items():
    #         for tname, tinfo in node_state.get("tiers", {}).items():
    #             new_tier_stats[tname] = new_tier_stats.get(tname, 0) + 1
    #             size_mb = tinfo.get("maxlen", 0) * (self.avg_content_size / (1024 * 1024))
    #             new_tier_sizes_mb[tname] = new_tier_sizes_mb.get(tname, 0) + size_mb

    #     # ===============================================================
    #     # 8) Save state to file
    #     # ===============================================================
    #     state = {
    #         # "old_node_carbon_intensity": self.node_carbon_intensity,
    #         # "old_node_carbon_intensity_total": sum(self.node_carbon_intensity.values()),

    #         "node_carbon_intensity": new_ci,
    #         "cache_hit_ratio": per_node_hit_ratio,
    #         "emb_raw":per_node_embodied,
    #         # "node_carbon_intensity_total": sum(new_ci.values()),

    #         "old_cache_size": old_cache_sizes,
    #         "old_cache_size_total": sum(old_cache_sizes.values()),
            
    #         "cache_size": new_cache_sizes,
    #         "cache_size_total": sum(new_cache_sizes.values()),

    #         # "old_tier_statistics": self.tier_statistics,
    #         "tier_statistics": new_tier_stats,

    #         # "old_tier_sizes_mb": self.tier_sizes_mb,
    #         "tier_sizes_mb": new_tier_sizes_mb,
            
    #         "old_per_node_tiers": old_per_node_tiers,
    #         "per_node_tiers": new_per_node_tiers,
    #     }


    #     with open(filepath, "wb") as f:
    #         pickle.dump(state, f)
    #     print(f"[💾] Saved network model state to {filepath}")

    #     jsonpath = filepath.with_suffix(".json")
    #     with open(jsonpath, "w") as jf:
    #         json.dump(state, jf, indent=2)
    #     print(f"[📄] JSON copy saved to {jsonpath}")

    #     # Apply new sizes & intensities for next iteration
    #     self.cache_size = new_cache_sizes
    #     self.node_carbon_intensity = new_ci
    #     self.tier_statistics = new_tier_stats
    #     self.tier_sizes_mb = new_tier_sizes_mb

    #     return str(filepath)


# import argparse
# from collections import Counter
# from icarus.registry import RESULTS_READER
# from icarus.util import Settings, config_logging


# def filter(resultset, strategy, topology, cache_size, alpha):
#     metric = "REPLICA_MONITOR"
#     cost_components = ["REPLICAS"]
#     # cost_components = ["MEAN", "DEPRECIATION", "BANDWIDTH", "READ_STORAGE", "WRITE_STORAGE", "ROUTERS", "LINKS", "PENALTY"]
#     other_strategies = resultset.filter({
#         "topology": {"name": topology},
#         "workload" :{"alpha":alpha},
#         "cache_placement": {"network_cache": cache_size},
#         metric: {}
#     })
#     output_file = "/home/lydia/icarus/examples/lce-vs-probcache/diff_change.csv"
#     with open(output_file, "a", encoding="utf-8") as f:
#         # f.write(f"\n--- {topology} ---\n")
#         for sub_metric in cost_components:
#             other_strategies_metric = {
#                 res[0].get("strategy").get("name"): res[1].get(metric).get(sub_metric)
#                 for res in other_strategies
#                 if res[1].get(metric).get(sub_metric) is not None
#             }

#             if "CL2SM" not in other_strategies_metric:
#                 f.write(f"[WARNING] CL2SM not found for sub_metric: {sub_metric}\n")
#                 continue
            
#             # strategy_name = "CL2SM"  # Change to your desired strategy
#             if strategy not in other_strategies_metric:
#                 f.write(f"[WARNING] {strategy} not found for sub_metric: {sub_metric}\n")
#                 continue
#             # Get the replicas dictionary for the chosen strategy
#             replicas_dict = other_strategies_metric[strategy]
#             # If it's a Tree, convert to dict
#             if hasattr(replicas_dict, "as_dict"):
#                 replicas_dict = replicas_dict.as_dict()

#             # Compute the distribution
#             replica_counts = Counter(replicas_dict.values())

#             # Write the distribution to the file
#             f.write(f"\nReplica distribution for {strategy} (sub_metric: {sub_metric}):\n")
#             f.write("Replicas,NumContents\n")
#             for n_replicas in sorted(replica_counts):
#                 f.write(f"{n_replicas},{replica_counts[n_replicas]}\n")
#             # total = sum(val for key, val in other_strategies_metric.items() if key != "CL2SM")
#             # avg = total/5
#             # return  ((other_strategies_metric["CL2SM"] - avg) / avg) * 100 

#             # percentage_change = {key: ((other_strategies_metric["CL2SM"] - val) / val) * 100 
#             #                     for (key, val) in other_strategies_metric.items() 
#             #                     if val != 0 and key != "CL2SM"
#             #                     }
#             # f.write(f"\n--- {sub_metric} ---\n")
#             # for key, val in percentage_change.items():
#             #     f.write(f"{key}: {val:.2f}%\n")

    


# def run(config, results, plotdir):
#     settings = Settings()
#     settings.read_from(config)
#     config_logging(settings.LOG_LEVEL)
#     resultset = RESULTS_READER[settings.RESULTS_FORMAT](results)

#     topologies = settings.TOPOLOGIES
#     cache_sizes = settings.NETWORK_CACHE
#     strategies = settings.STRATEGIES
#     alphas = settings.ALPHA

#     topology = "GEANT"
#     cache_size = 0.005
#     alpha = 1.2
#     counter =0
#     avg = 0
#     for strategy in strategies:
#         for topology in topologies:
#             for cache_size in cache_sizes:
#                 for alpha in alphas:
#                     counter += 1
#                     filter(resultset, strategy, topology, cache_size, alpha)
#     # print(counter)
#     # filter(resultset, topology, cache_size, alpha)

#     # print(avg/counter)

# def main():
#     parser = argparse.ArgumentParser(__doc__)
#     parser.add_argument("-r", "--results", dest="results", help="the results file", required=True)
#     parser.add_argument(
#         "-o",
#         "--output",
#         dest="output",
#         help="the output directory where plots will be saved",
#         required=True,
#     )
#     parser.add_argument("config", help="the configuration file")
#     args = parser.parse_args()
#     run(args.config, args.results, args.output)

# if __name__ == "__main__":
#     main()

# import matplotlib.pyplot as plt
# import numpy as np

# # Style and legend dictionaries (as before)
# MATPLOTLIB_COLOR_TO_HEX = {
#     "b": "#1f77b4",
#     "g": "#2ca02c",
#     "r": "#d62728",
#     "c": "#17becf",
#     "m": "#9467bd",
#     "k": "#000000",
# }

# STRATEGY_STYLE = {
#     "LCE": "b-v",
#     "LCD": "g-o",
#     "PROB_CACHE": "c-*",
#     "RAND_CHOICE": "k-d",
#     "CL2SM": "r-s",
#     "CPCache": "ko-",
#     "CL4M": "m-->",
# }

# STRATEGY_LEGEND = {
#     "LCE": "LCE",
#     "LCD": "LCD",
#     "PROB_CACHE": "ProbCache",
#     "RAND_CHOICE": "Random",
#     "CL2SM": "CL2SM",
#     "CPCache": "CPCache",
#     "CL4M": "CL4M",
# }

# distributions = {
#     "CL2SM": {1: 4520, 2: 66, 3: 57, 4: 4, 5: 6, 6: 1, 7: 1, 8: 4, 9: 1, 10: 2},
#     "LCE": {2: 192, 3: 812, 4: 1222, 5: 1112, 6: 964, 7: 350, 8: 6, 9: 2, 10: 1, 11: 1},
#     "LCD": {2: 4590, 3: 45, 4: 11, 5: 6, 6: 4, 7: 3, 8: 1, 9: 1, 10: 1},
#     "PROB_CACHE": {1: 3373, 2: 1191, 3: 90, 4: 8, 6: 2, 7: 1, 8: 2, 9: 2},
#     "CL4M": {2: 4617, 3: 31, 4: 5, 5: 2, 6: 1, 8: 2, 9: 2, 10: 2},
#     "CPCache": {1: 183, 2: 792, 3: 1206, 4: 1110, 5: 974, 6: 368, 7: 20, 8: 6, 9: 1, 10: 2},
# }

# def parse_style(style):
#     color = None
#     linestyle = '-'
#     marker = None

#     # Color at start
#     if style and style[0] in MATPLOTLIB_COLOR_TO_HEX:
#         color = MATPLOTLIB_COLOR_TO_HEX[style[0]]
#         style = style[1:]
#     # Linestyle
#     for ls in ['--', '-.', '-', ':']:
#         if style.startswith(ls):
#             linestyle = ls
#             style = style[len(ls):]
#             break
#     # Marker
#     if style:
#         marker = style[0]
#     return color, linestyle, marker

# fontsize = 24

# all_replica_counts = sorted({k for d in distributions.values() for k in d})


# # Set global rcParams
# plt.rcParams["text.usetex"] = False
# plt.rcParams["figure.figsize"] = (12, 5)
# plt.rcParams["legend.fontsize"] = 16
# plt.rcParams["axes.labelsize"] = 24
# plt.rcParams["axes.titlesize"] = 24
# plt.rcParams["xtick.labelsize"] = 24
# plt.rcParams["ytick.labelsize"] = 24
# plt.rcParams["lines.linewidth"] = 1.5
# plt.rcParams["lines.markersize"] = 12

# # ... (define your STRATEGY_STYLE, STRATEGY_LEGEND, distributions, and parse_style as before) ...

# fig, ax = plt.subplots()

# for strategy, dist in distributions.items():
#     if strategy not in STRATEGY_STYLE:
#         continue
#     style = STRATEGY_STYLE[strategy]
#     color, linestyle, marker = parse_style(style)
#     y = [dist.get(x, 0) for x in all_replica_counts]
#     ax.plot(
#         all_replica_counts,
#         y,
#         linestyle=linestyle,
#         marker=marker,
#         color=color,
#         label=STRATEGY_LEGEND.get(strategy, strategy),
#     )

# ax.set_title("Replica Distribution Across Caching Strategies")
# ax.set_xlabel("Number of Replicas")
# ax.set_ylabel("Number of Contents")
# ax.set_xticks(all_replica_counts)
# ax.set_xticklabels([str(x) for x in all_replica_counts])
# ax.grid(True, which='both', linestyle='--', linewidth=0.8)
# ax.legend(loc="best")
# plt.tight_layout()
# plt.savefig("replica_distribution.png", bbox_inches="tight")
# plt.close(fig)



# import matplotlib.pyplot as plt
# import numpy as np

# # Example data (fill with your real distributions)
# strategies = ['CL2SM', 'LCE', 'LCD', 'PROB_CACHE', 'CL4M', 'CPCache']
# replica_counts = sorted({k for d in distributions.values() for k in d})
# bar_width = 0.13
# x = np.arange(len(replica_counts))

# plt.figure(figsize=(14, 7))
# for idx, strategy in enumerate(strategies):
#     y = [distributions[strategy].get(rc, 0) for rc in replica_counts]
#     plt.bar(x + idx*bar_width, y, width=bar_width, label=strategy)

# plt.xlabel('Number of Replicas')
# plt.ylabel('Number of Contents')
# plt.title('Replica Distribution Comparison')
# plt.xticks(x + bar_width*len(strategies)/2, replica_counts)
# plt.legend()
# plt.tight_layout()
# plt.savefig('replica_distribution_grouped_bar.png', dpi=300)
# plt.close()


# import pandas as pd

# stats = {}
# for strategy, dist in distributions.items():
#     data = []
#     for rc, count in dist.items():
#         data.extend([rc] * count)
#     stats[strategy] = {
#         'mean': np.mean(data),
#         'median': np.median(data),
#         'min': np.min(data),
#         'max': np.max(data),
#         'std': np.std(data),
#     }
# df = pd.DataFrame(stats).T
# print(df)

# import matplotlib.pyplot as plt
# import numpy as np

# plt.figure(figsize=(10, 6))
# for strategy, dist in distributions.items():
#     xs = sorted(dist)
#     ys = [dist[x] for x in xs]
#     cdf = np.cumsum(ys) / sum(ys)
#     plt.step(xs, cdf, where='post', label=strategy)

# plt.xlabel('Number of Replicas')
# plt.ylabel('Cumulative Fraction of Contents')
# plt.title('CDF of Replica Counts by Strategy')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('replica_distribution_cdf.png', dpi=300)
# plt.close()

# import csv

# # Path to your original CSV file
# input_file_path = '/home/lydia/icarus/examples/lce-vs-probcache/cost_log.csv'
# # Path to the output CSV file with the specified lines removed
# output_file_path = '/home/lydia/icarus/examples/lce-vs-probcache/cleaned_log.csv'

# # Read the input CSV file and write to the output CSV file
# with open(input_file_path, mode='r', newline='') as infile, open(output_file_path, mode='w', newline='') as outfile:
#     reader = csv.reader(infile)
#     writer = csv.writer(outfile)

#     # Convert the rows into a list for easy manipulation
#     rows = list(reader)
#     skip_next = False

#     # Iterate over the rows
#     for i in range(len(rows)):
#         if skip_next:
#             # Skip this line because the previous line had only zeros and this is the "next line"
#             skip_next = False
#             continue

#         # Check if the current row contains only 0.0 values
#         if all(float(val) == 0.0 for val in rows[i]):
#             # Mark the next line to be skipped
#             skip_next = True
#         else:
#             # Write the current row to the output file
#             writer.writerow(rows[i])

# # Path to your CSV file
# input_file_path = '/home/lydia/icarus/examples/lce-vs-probcache/cleaned_log.csv'
# # Prepare to store sums of even and odd rows
# even_row_sums = [0.0] * 5  # Assuming there are 5 columns
# odd_row_sums = [0.0] * 5

# # Read the input CSV file
# with open(input_file_path, mode='r', newline='') as infile:
#     reader = csv.reader(infile)
#     # Iterate through rows, determining if row is even or odd based on index
#     for index, row in enumerate(reader):
#         if index % 2 == 0:  # even index, impair row (starting index is 0)
#             even_row_sums = [even_row_sums[i] + float(row[i]) for i in range(len(row))]
#         else:  # odd index, pair row
#             odd_row_sums = [odd_row_sums[i] + float(row[i]) for i in range(len(row))]

# # Output the results
# print("Sum of impair rows:", even_row_sums)
# print("Sum of pair rows:", odd_row_sums)

# Sum of impair rows: [400.86558480000207, 0.20153089348184422, 0.0037562000000000563, 17.73058155626369, 3.0809869391700696e-05]
# Sum of pair rows: [298.3211496000018, 0.10863016621342868, 0.0024334000000001943, 1.0622633289073413, 2.4433321106098796e-06]

# "Bandwidth Cost", "Transmission Cost", "Penalty Cost", "Depreciation Cost", "Storage Cost
# >
# ESTIMATED [68.33417519999988, 0.03176995811119611, 0.00043240000000000303, 1.674207100615174, 3.7984301221288554e-06]
# REAL [102.51627000000003, 0.03733010369735005, 0.0007745999999999781, 0.7301204541275115, 2.2987560739126874e-06]
# 40, 14, 56, 78, 49

# cost is not for it
# ESTIMATED [400.86558480000207, 0.20153089348184422, 0.0037562000000000563, 17.73058155626369, 3.0809869391700696e-05]
# REAL [298.3211496000018, 0.10863016621342868, 0.0024334000000001943, 1.0622633289073413, 2.4433321106098796e-06]
# 29, 66, 42, 177, 170

# import matplotlib.pyplot as plt
# import numpy as np

# # Cost categories and their respective colors and hatch patterns
# categories = ["Bandwidth", "Transmission", "Penalty", "Depreciation", "Storage"]
# colors = ['#1F77B4', '#D62728', '#E377C2', '#FF7F0E', '#2CA02C']
# hatches = ['/', '\\', '|', '-', '+', 'x']  # Different hatch patterns for each category

# # # Estimated and real costs for demonstration
# # # estimated_costs = [68.33417519999988, 0.03176995811119611, 0.00043240000000000303, 1.674207100615174, 3.7984301221288554e-06]
# # # real_costs = [102.51627000000003, 0.03733010369735005, 0.0007745999999999781, 0.7301204541275115, 2.2987560739126874e-06]

# estimated_costs = [400.86558480000207, 0.20153089348184422, 0.0037562000000000563, 17.73058155626369, 3.0809869391700696e-05]
# real_costs = [298.3211496000018, 0.10863016621342868, 0.0024334000000001943, 1.0622633289073413, 2.4433321106098796e-06]

# # Setup for the plot
# fig, ax = plt.subplots()

# # Positions for the groups on the x-axis
# positions = np.arange(2)  # positions for 'Estimated' and 'Real'

# # Stack each category cost on the respective group
# bottom_estimated = 0
# bottom_real = 0

# for i, (color, hatch) in enumerate(zip(colors, hatches)):
#     # Add bars for estimated costs
#     ax.bar(positions[0], estimated_costs[i], color=color, hatch=hatch, width=0.4, bottom=bottom_estimated, edgecolor='black', label=categories[i])
#     bottom_estimated += estimated_costs[i]
    
#     # Add bars for real costs
#     ax.bar(positions[1], real_costs[i], color=color, hatch=hatch, width=0.4, bottom=bottom_real, edgecolor='black')
#     bottom_real += real_costs[i]

# # Add some text for labels, title and custom x-axis tick labels
# ax.set_ylabel('Costs')
# ax.set_title('Stacked Costs by Type')
# ax.set_xticks(positions)
# ax.set_xticklabels(['Estimated', 'Real'])
# ax.legend(title="Cost Categories", bbox_to_anchor=(1.05, 1), loc='upper left')

# fig.tight_layout()

# # Save the figure
# plt.savefig('detailed_stacked_cost_comparison.jpg', format='jpg', dpi=300)  # Save as JPG file with high resolution
# plt.close(fig)  # Close the plot figure to free up memory

# def init_fn(icr_candidates, params, metrics, settings):
#     print("Initial solution")
#     cache_budget = params["workload"]["n_contents"] * params["cache_placement"]["network_cache"] 
    
#     results_file = EXAMPLES_DIR / "results.pickle"
#     config_file = EXAMPLES_DIR / "paes_config.py"
#     exp_pkl = EXAMPLES_DIR / "exp.pkl"
    
#     # --- Step 1: Build an experiment with init placement ---
#     exp = build_experiment(
#         icr_candidates=icr_candidates,
#         params=params,
#         metrics=metrics,
#         allocations=[],
#         tiers_per_node=None,
#         cache_placement = "UNIFORM"
#     )
#     if hasattr(exp, "to_dict"):
#         exp = exp.to_dict()

#     with open(exp_pkl, "wb") as f:
#         pickle.dump([exp], f)

#     with open(config_file, "w") as f:
#         f.write("from collections import deque\n")
#         f.write("import pickle\n")
#         f.write(f"LOG_LEVEL = {repr(str(settings.LOG_LEVEL))}\n")
#         f.write(f"CACHING_GRANULARITY = {repr(str(settings.CACHING_GRANULARITY))}\n")
#         f.write(f"RESULTS_FORMAT = {repr(str(settings.RESULTS_FORMAT))}\n")
#         f.write(f"PARALLEL_EXECUTION = {settings.PARALLEL_EXECUTION}\n")
#         f.write(f"N_REPLICATIONS = {settings.N_REPLICATIONS}\n")
#         f.write(f"N_PERIODS = {settings.N_PERIODS}\n")
#         f.write("EXPERIMENT_QUEUE = deque()\n")
#         f.write(f"EXPERIMENT_QUEUE.extend(pickle.load(open({repr(str(exp_pkl))}, 'rb')))\n")
    
#     # --- Step 2: Run Icarus once so cache_allocations get filled ---
#     run(str(config_file), str(results_file), {})

#     # --- Step 3: Read back the allocations from results.pickle ---
#     with open(results_file, "rb") as f:
#         results = pickle.load(f)
    
#     # for item in results[0]:
#     #     print(f"item:{item}")
#     _, metrics = results[0]  # params, metrics
#     alloc_dict = metrics.get("cache_allocations", {})
    
#     allocs = [alloc_dict.get(v, 0) for v in icr_candidates]
#     tiers_per_node = {node: copy.deepcopy(params["cache_policy"]["tiers"]) for node in icr_candidates}


#     print(f"allocs:{allocs}")
    
#     return {
#         "allocations": allocs,
#         "cache_budget": cache_budget,
#         "icr_candidates": icr_candidates,
#         "tiers_per_node": tiers_per_node,
#     }

# # --- Mutation function ---
# def mutate_fn(sol):
#     logger.info("Mutate solution")
#     new_sol = copy.deepcopy(sol)
#     allocs = new_sol["allocations"][:]
#     n_nodes = len(allocs)
    
#     # perturb one node allocation
#     for _ in range(random.randint(3, 8)):
#         idx = random.randrange(n_nodes)
#         delta = random.randint(-30, 30)   # larger mutation
#         allocs[idx] = max(0, allocs[idx] + delta)

#     total = sum(allocs)
#     if total > 0:
#         # scale back to original cache budget
#         scaled = [a / total * sol["cache_budget"] for a in allocs]
#         floored = [int(math.floor(a)) for a in scaled]

#         # FIX: remainder must be an int
#         remainder = int(sol["cache_budget"] - sum(floored))

#         # distribute leftover to largest fractional parts
#         fractions = sorted(
#             enumerate([a - f for a, f in zip(scaled, floored)]),
#             key=lambda x: x[1],
#             reverse=True
#         )
#         for i, _ in fractions[:remainder]:
#             floored[i] += 1
#         allocs = floored

#     new_sol["allocations"] = allocs
#     return new_sol

import os
import re

folder = "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/network_states"

pattern = re.compile(r"^exp(\d+)_.*GEANT.*")

# Rename in descending order to avoid overwriting
for filename in sorted(os.listdir(folder), reverse=True):
    match = pattern.match(filename)
    if match:
        exp_num = 1
        new_name = filename.replace(
            f"exp{exp_num}_",
            f"exp{4}_",
            1
        )

        os.rename(
            os.path.join(folder, filename),
            os.path.join(folder, new_name)
        )

# import os
# import re
# import pickle

# folder = "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/network_states"

# pattern = re.compile(r"^exp(\d+)_GEANT_.*\.json$")

# for filename in os.listdir(folder):
#     match = pattern.match(filename)
#     if not match:
#         continue

#     exp_num = int(match.group(1))

#     # 🔴 STOP at exp30
#     if exp_num <= 30:
#         continue

#     pkl_path = os.path.join(folder, filename)

#     # Load
#     with open(pkl_path, "r") as f:
#         print(f)
#         data = json.load(f)

#     # Update content
#     print(data["alpha"])
#     data["alpha"] = 1.2

#     # Save back
#     with open(pkl_path, "w") as f:
#         print(f)
#         json.dump(data, f, indent=2)
#         # json.dump(loaded_data, txt_file, indent=2)

#     print(f" {filename}")

# import json
# import pickle

# class ResultSet:
#     def __init__(self, results):
#         self.results = results

# try:
#     # Read JSON
#     with open(
#         '/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/network_states/GEA/exp32_GEANT_p1.json',
#         'r'
#     ) as file:
#         loaded_data = json.load(file)
#         print("Loaded JSON successfully!")

#         results = loaded_data   # mirror your logic

#     # Write PKL
#     with open(
#         '/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/network_states/exp1_GEANT_p1_read.pkl',
#         'wb'
#     ) as pkl_file:
#         pickle.dump(results, pkl_file)

#     print("Results written to PKL successfully!")

# except (json.JSONDecodeError, FileNotFoundError, pickle.PickleError) as e:
#     print("Error converting JSON to PKL:", e)
