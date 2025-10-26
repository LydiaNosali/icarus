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

import matplotlib.pyplot as plt
import numpy as np

# Style and legend dictionaries (as before)
MATPLOTLIB_COLOR_TO_HEX = {
    "b": "#1f77b4",
    "g": "#2ca02c",
    "r": "#d62728",
    "c": "#17becf",
    "m": "#9467bd",
    "k": "#000000",
}

STRATEGY_STYLE = {
    "LCE": "b-v",
    "LCD": "g-o",
    "PROB_CACHE": "c-*",
    "RAND_CHOICE": "k-d",
    "CL2SM": "r-s",
    "CPCache": "ko-",
    "CL4M": "m-->",
}

STRATEGY_LEGEND = {
    "LCE": "LCE",
    "LCD": "LCD",
    "PROB_CACHE": "ProbCache",
    "RAND_CHOICE": "Random",
    "CL2SM": "CL2SM",
    "CPCache": "CPCache",
    "CL4M": "CL4M",
}

distributions = {
    "CL2SM": {1: 4520, 2: 66, 3: 57, 4: 4, 5: 6, 6: 1, 7: 1, 8: 4, 9: 1, 10: 2},
    "LCE": {2: 192, 3: 812, 4: 1222, 5: 1112, 6: 964, 7: 350, 8: 6, 9: 2, 10: 1, 11: 1},
    "LCD": {2: 4590, 3: 45, 4: 11, 5: 6, 6: 4, 7: 3, 8: 1, 9: 1, 10: 1},
    "PROB_CACHE": {1: 3373, 2: 1191, 3: 90, 4: 8, 6: 2, 7: 1, 8: 2, 9: 2},
    "CL4M": {2: 4617, 3: 31, 4: 5, 5: 2, 6: 1, 8: 2, 9: 2, 10: 2},
    "CPCache": {1: 183, 2: 792, 3: 1206, 4: 1110, 5: 974, 6: 368, 7: 20, 8: 6, 9: 1, 10: 2},
}

def parse_style(style):
    color = None
    linestyle = '-'
    marker = None

    # Color at start
    if style and style[0] in MATPLOTLIB_COLOR_TO_HEX:
        color = MATPLOTLIB_COLOR_TO_HEX[style[0]]
        style = style[1:]
    # Linestyle
    for ls in ['--', '-.', '-', ':']:
        if style.startswith(ls):
            linestyle = ls
            style = style[len(ls):]
            break
    # Marker
    if style:
        marker = style[0]
    return color, linestyle, marker

fontsize = 24

all_replica_counts = sorted({k for d in distributions.values() for k in d})


# Set global rcParams
plt.rcParams["text.usetex"] = False
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["axes.labelsize"] = 24
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams["lines.markersize"] = 12

# ... (define your STRATEGY_STYLE, STRATEGY_LEGEND, distributions, and parse_style as before) ...

fig, ax = plt.subplots()

for strategy, dist in distributions.items():
    if strategy not in STRATEGY_STYLE:
        continue
    style = STRATEGY_STYLE[strategy]
    color, linestyle, marker = parse_style(style)
    y = [dist.get(x, 0) for x in all_replica_counts]
    ax.plot(
        all_replica_counts,
        y,
        linestyle=linestyle,
        marker=marker,
        color=color,
        label=STRATEGY_LEGEND.get(strategy, strategy),
    )

ax.set_title("Replica Distribution Across Caching Strategies")
ax.set_xlabel("Number of Replicas")
ax.set_ylabel("Number of Contents")
ax.set_xticks(all_replica_counts)
ax.set_xticklabels([str(x) for x in all_replica_counts])
ax.grid(True, which='both', linestyle='--', linewidth=0.8)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig("replica_distribution.png", bbox_inches="tight")
plt.close(fig)



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

