import argparse
import os
from icarus.registry import RESULTS_READER
from icarus.util import Settings, config_logging


def filter(
        resultset, cache_size, alpha, strategy
    ):
    # lce_filtered = resultset.filter({
    #     "strategy": {"name": "LCE"},
    #     "workload" :{"alpha":alpha},
    #     "CHRCP": {}
    # })
    
    # # Step 2: Create a dictionary of LCE CHRCP values for normalization
    # lce_cost = {
    #     res[0].get("cache_placement").get("network_cache"): res[1].get("COST").get("MEAN")
    #     for res in lce_filtered
    #     if res[1].get("COST").get("MEAN") is not None
    # }

    # alpha_filtered = resultset.filter({
    #     "workload" :{"alpha":alpha},
    #     "CHRCP": {}
    # })

    # # Step 3: Normalize the resultset based on LCE CHRCP values
    # for entry, metrics in alpha_filtered:
    #     cache_size = entry.get("cache_placement", {}).get("network_cache")
    #     if cache_size in lce_cost and metrics.get("CHRCP", {}).get("MEAN") is not None:
    #         normalized_value = metrics["CHRCP"]["MEAN"] / lce_cost[cache_size]
    #         metrics["CHRCP"]["MEAN"] = normalized_value
    
    metric = "CACHE_HIT_RATIO"
    # metric = "COST"
    # metric = "CHRCP"
    strat_filtered = resultset.filter({
        "strategy": {"name": strategy},
        "workload" :{"alpha":alpha},
        "cache_placement": {"network_cache": cache_size},
        metric: {}
    })
    strat_cost = {
        res[0].get("cache_placement").get("network_cache"): res[1].get(metric).get("MEAN")
        for res in strat_filtered
        if res[1].get(metric).get("MEAN") is not None
    }
    cost_filtered = resultset.filter({
        "strategy": {"name": "COST"},
        "workload" :{"alpha":alpha},
        "cache_placement": {"network_cache": cache_size},
        metric: {}
    })
    cost_metric = {
        res[0].get("cache_placement").get("network_cache"): res[1].get(metric).get("MEAN")
        for res in cost_filtered
        if res[1].get(metric).get("MEAN") is not None
    }
    # Calculate percentage decrease for each key
    percentage_decrease = {key: ((strat_cost[key] - cost_metric[key]) / strat_cost[key]) * 100 for key in strat_cost}
    percentage_increase = {key: ((cost_metric[key] - strat_cost[key]) / cost_metric[key]) * 100 for key in cost_metric}

    # Print the results
    for key, decrease in percentage_increase.items():
        print(f"Decrease for key {key}: {decrease:.2f}%")
    


def run(config, results, plotdir):
    settings = Settings()
    settings.read_from(config)
    config_logging(settings.LOG_LEVEL)
    resultset = RESULTS_READER[settings.RESULTS_FORMAT](results)
    # Create dir if not existsing
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    # Parse params from settings

    # cache_items = settings.NETWORK_CACHE
    original_list = []
    for entry, metrics in resultset:
        cache_size = entry.get("cache_placement", {})["network_cache"]
        workload = entry.get("workload", {}).get("n_contents")
        normalized_value = int(cache_size * workload / 13)
        entry.get("cache_placement", {})["network_cache"] = normalized_value
        original_list.append(normalized_value)
    
    cache_sizes = list(dict.fromkeys(original_list))
    strategies = settings.STRATEGIES
    alphas = settings.ALPHA
    # Plot graphs
    for strategy in strategies:
        print(f"strategy:{strategy}")
        for cache_size in cache_sizes:
            print(f"size:{cache_size}")
            for alpha in alphas:
                print(f"alpha:{alpha}")
                filter(
                    resultset, cache_size, alpha, strategy
                )

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "-r", "--results", dest="results", help="the results file", required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="the output directory where plots will be saved",
        required=True,
    )
    parser.add_argument("config", help="the configuration file")
    args = parser.parse_args()
    run(args.config, args.results, args.output)

if __name__ == "__main__":
    main()
