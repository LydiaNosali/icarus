"""Plot results read from a result set
"""
import os
import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from icarus.util import Settings, config_logging
from icarus.results import plot_lines, plot_bar_chart
from icarus.registry import RESULTS_READER


# Logger object
logger = logging.getLogger("plot")

# These lines prevent insertion of Type 3 fonts in figures
# Publishers don't want them
plt.rcParams["ps.useafm"] = True
plt.rcParams["pdf.use14corefonts"] = True

# If True text is interpreted as LaTeX, e.g. underscore are interpreted as
# subscript. If False, text is interpreted literally
plt.rcParams["text.usetex"] = False

# Aspect ratio of the output figures
plt.rcParams["figure.figsize"] = 8, 5

# Size of font in legends
LEGEND_SIZE = 14

# Line width in pixels
LINE_WIDTH = 1.5

# Plot
PLOT_EMPTY_GRAPHS = True

# This dict maps strategy names to the style of the line to be used in the plots
# Off-path strategies: solid lines
# On-path strategies: dashed lines
# No-cache: dotted line
STRATEGY_STYLE = {
    "LCD": "g-o",
    "RAND_CHOICE": "m--<",
    "COST": "r-^",
    "LCE": "b--p",
    "PROB_CACHE": "c--<",
    # "COST_CACHE": "m-^",
    # "RAND_BERNOULLI": "g--*",
    # "NO_CACHE": "k:o",
    # "OPTIMAL": "k-o",
    
}

# This dict maps name of strategies to names to be displayed in the legend
STRATEGY_LEGEND = {
    "LCE": "LCE",
    "LCD": "LCD",
    "PROB_CACHE": "ProbCache",
    "COST": "Cost",
    "RAND_CHOICE": "Random (choice)",
    # "HR_SYMM": "HR Symm",
    # "HR_ASYMM": "HR Asymm",
    # "HR_MULTICAST": "HR Multicast",
    # "HR_HYBRID_AM": "HR Hybrid AM",
    # "HR_HYBRID_SM": "HR Hybrid SM",
    # "CL4M": "CacheLessForMore",
    # "RAND_BERNOULLI": "Random (Bernoulli)",
    # "NO_CACHE": "No caching",
    # "OPTIMAL": "Optimal",
}

# Color and hatch styles for bar charts of cache hit ratio and link load vs topology
STRATEGY_BAR_COLOR = {
    "LCE": "k",
    "RAND_CHOICE" :"0.7",
    "LCD": "0.4",
    "PROB_CACHE": "0.5",
    "COST": "0.6",
    # "CL4M": "0.6",
    # "NO_CACHE": "0.5",
    # "HR_ASYMM": "0.6",
    # "HR_SYMM": "0.7",   
}

STRATEGY_BAR_HATCH = {
    "LCE": None,
    "RAND_CHOICE" : "x",
    "LCD": "//",
    "PROB_CACHE": "\\",
    "COST": "+",
    # "NO_CACHE": "x",
    # "HR_ASYMM": "+",
    # "HR_SYMM": "\\",
    # "CL4M": "x",
}


def plot_cache_hits_vs_cache_size(
    resultset, topology, alpha, cache_size_range, strategies, plotdir
):
    desc = {}
    if "NO_CACHE" in strategies:
        strategies.remove("NO_CACHE")
    desc["title"] = "Cache hit ratio: T={} A={}".format(topology, alpha)
    desc["xlabel"] = "Cache to population ratio"
    desc["ylabel"] = "Cache hit ratio"
    desc["xscale"] = "log"
    desc["xparam"] = ("cache_placement", "network_cache")
    desc["xvals"] = cache_size_range
    desc["filter"] = {
        "topology": {"name": topology},
        "workload": {"name": "STATIONARY", "alpha": alpha},
    }
    desc["ymetrics"] = [("CACHE_HIT_RATIO", "MEAN")] * len(strategies)
    desc["ycondnames"] = [("strategy", "name")] * len(strategies)
    desc["ycondvals"] = strategies
    desc["errorbar"] = True
    desc["legend_loc"] = "upper left"
    desc["line_style"] = STRATEGY_STYLE
    desc["legend"] = STRATEGY_LEGEND
    desc["plotempty"] = PLOT_EMPTY_GRAPHS
    plot_lines(
        resultset,
        desc,
        "CACHE_HIT_RATIO_T={}@A={}.jpg".format(topology, alpha),
        plotdir,
    )

def plot_cost_vs_cache_size(
    resultset, topology, alpha, cache_size_range, strategies, plotdir
):
    desc = {}
    desc["title"] = "Cost: T={} A={}".format(topology, alpha)
    desc["xlabel"] = "Cache to population ratio"
    desc["ylabel"] = "Cost"
    desc["xscale"] = "log"
    desc["xparam"] = ("cache_placement", "network_cache")
    desc["xvals"] = cache_size_range
    desc["filter"] = {
        "topology": {"name": topology},
        "workload": {"name": "STATIONARY", "alpha" : alpha},
    }
    desc["ymetrics"] = [("COST", "MEAN")] * len(strategies)
    desc["ycondnames"] = [("strategy", "name")] * len(strategies)
    desc["ycondvals"] = strategies
    desc["metric"] = ("COST", "MEAN")
    desc["errorbar"] = True
    desc["legend_loc"] = "upper right"
    desc["line_style"] = STRATEGY_STYLE
    desc["legend"] = STRATEGY_LEGEND
    desc["plotempty"] = PLOT_EMPTY_GRAPHS
    plot_lines(
        resultset, desc, "COST_T={}@A={}.jpg".format(topology, alpha), plotdir
    )

def plot_chrcp_vs_cache_size(
    resultset, topology, alpha, cache_size_range, strategies, plotdir):
    # Step 1: Filter results for the LCE strategy, specified topology, and CHRCP metric
    lce_filtered = resultset.filter({
        "topology": {"name": topology},
        "strategy": {"name": "LCE"},
        "CHRCP": {}
    })
    
    # Step 2: Create a dictionary of LCE CHRCP values for normalization
    lce_cost = {
        res[0].get("cache_placement").get("network_cache"): res[1].get("COST").get("MEAN")
        for res in lce_filtered
        if res[1].get("COST").get("MEAN") is not None
    }
    
    if not lce_cost:
        logger.error("No LCE CHRCP values found for normalization.")
        return
    
    # Step 3: Normalize the resultset based on LCE CHRCP values
    for entry, metrics in resultset:
        cache_size = entry.get("cache_placement", {}).get("network_cache")
        if cache_size in lce_cost and metrics.get("CHRCP", {}).get("MEAN") is not None:
            normalized_value = metrics["CHRCP"]["MEAN"] / lce_cost[cache_size]
            metrics["CHRCP"]["MEAN"] = normalized_value
    
    # Step 4: Plot the normalized CHRCP results
    desc = {
        "title": f"CHRCP: T={topology} A={alpha}",
        "xlabel": "Cache to population ratio",
        "ylabel": "CHRCP (relative to LCE)",
        "xscale": "log",
        "xparam": ("cache_placement", "network_cache"),
        "xvals": cache_size_range,
        "filter": {
            "topology": {"name": topology},
            "workload": {"name": "STATIONARY", "alpha": alpha},
        },
        "ymetrics": [("CHRCP", "MEAN")] * len(strategies),
        "ycondnames": [("strategy", "name")] * len(strategies),
        "ycondvals": strategies,
        "metric": ("CHRCP", "MEAN"),
        "errorbar": True,
        "legend_loc": "upper right",
        "line_style": STRATEGY_STYLE,
        "legend": STRATEGY_LEGEND,
        "plotempty": PLOT_EMPTY_GRAPHS,
    }
    
    plot_lines(
        resultset, desc, f"CHRCP_T={topology}@A={alpha}.jpg", plotdir
    )

def plot_latency_vs_cache_size(
    resultset, topology, alpha, cache_size_range, strategies, plotdir
):
    desc = {}
    desc["title"] = "Latency: T={} A={}".format(topology, alpha)
    desc["xlabel"] = "Cache to population ratio"
    desc["ylabel"] = "Latency"
    desc["xscale"] = "log"
    desc["xparam"] = ("cache_placement", "network_cache")
    desc["xvals"] = cache_size_range
    desc["filter"] = {
        "topology": {"name": topology},
        "workload": {"name": "STATIONARY", "alpha": alpha},
    }
    desc["ymetrics"] = [("LATENCY", "MEAN")] * len(strategies)
    desc["ycondnames"] = [("strategy", "name")] * len(strategies)
    desc["ycondvals"] = strategies
    desc["metric"] = ("LATENCY", "MEAN")
    desc["errorbar"] = True
    desc["legend_loc"] = "upper right"
    desc["line_style"] = STRATEGY_STYLE
    desc["legend"] = STRATEGY_LEGEND
    desc["plotempty"] = PLOT_EMPTY_GRAPHS
    plot_lines(
        resultset, desc, "LATENCY_T={}@A={}.jpg".format(topology, alpha), plotdir
    )

def plot_cost_components_vs_cache_size(
    resultset, topology, alpha, cache_size_range, strategies, plotdir
):
    """
    Plot cost components for each strategy as a stacked bar plot with strategy names under each bar.
    """
    # Cost component names in the result set
    cost_components = ["DEPRECIATION", "BANDWIDTH", "READ_STORAGE", "WRITE_STORAGE", "ROUTERS", "LINKS", "PENALTY"]
    num_components = len(cost_components)
    
    # Prepare for plotting
    fig, ax = plt.subplots(figsize=(10, 6))  # Increase figure size
    bar_width = 0.15  # Width of each strategy's bar
    bar_spacing = 0.05  # Extra space between groups of bars
    total_bars_per_group = len(strategies) * (bar_width + bar_spacing)
    
    # Generate positions for each bar, spacing them based on both cache sizes and strategies
    x_positions = []
    for i, cache_size in enumerate(cache_size_range):
        for j, strategy in enumerate(strategies):
            x_positions.append(i * (total_bars_per_group + 0.2) + j * (bar_width + bar_spacing))
    
    x_positions = np.array(x_positions)
    
    # Define color and hatch styles for each cost component
    cost_colors = ['#FF7F0E', '#1F77B4', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2']
    cost_hatches = ['/', '\\', '|', '-', '+', 'x', 'o']
    
    # Plot bars for each strategy and component
    for i, strategy in enumerate(strategies):
        bottom = np.zeros(len(cache_size_range))  # Initialize for stacking bars

        for j, component in enumerate(cost_components):
            data = []
            for cache_size in cache_size_range:
                filtered = resultset.filter({
                    "topology": {"name": topology},
                    "cache_placement": {"network_cache": cache_size},
                    "strategy": {"name": strategy},
                    "workload" :{"name": "STATIONARY", "alpha": alpha},
                })
                
                cost = filtered[0][1]['COST'].get(component, 0) if len(filtered) > 0 else 0
                data.append(cost)
            
            ax.bar(
                x_positions[i::len(strategies)], data, bar_width,
                bottom=bottom,
                color=cost_colors[j],
                hatch=cost_hatches[j]
            )
            bottom += np.array(data)

    # Set labels, title, ticks, and legends
    ax.set_xlabel('Cache Proportion and Strategy', fontsize=14)
    ax.set_ylabel('Cost', fontsize=14)
    ax.set_title('Cost Components per Cache Size and Strategy', fontsize=16)
    
    # Add cache size and strategy labels as x-axis labels
    xtick_labels = []
    for cache_size in cache_size_range:
        for strategy in strategies:
            xtick_labels.append(f'{strategy}\n(Cache {cache_size})')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xtick_labels, fontsize=10, rotation=45, ha="right")
    
    # Add gridlines
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Add a legend for the cost components only
    handles = [plt.Rectangle((0,0),1,1, color=cost_colors[i], hatch=cost_hatches[i]) for i in range(num_components)]
    ax.legend(handles, cost_components, loc='upper right', fontsize=10, title="Cost Components")

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir, f"COST_COMPONENTS_T={topology}@A={alpha}.jpg"), bbox_inches='tight')
    # plt.show()cache_size_rangecache_size_range

def plot_cache_hits_vs_topology(
    resultset, alpha, cache_size, topology_range, strategies, plotdir
):
    """
    Plot bar graphs of cache hit ratio for specific values of alpha and cache
    size for various topologies.

    The objective here is to show that our algorithms works well on all
    topologies considered
    """
    if "NO_CACHE" in strategies:
        strategies.remove("NO_CACHE")
    desc = {}
    desc["title"] = "Cache hit ratio: A={} C={}".format(alpha, cache_size)
    desc["ylabel"] = "Cache hit ratio"
    desc["xparam"] = ("topology", "name")
    desc["xvals"] = topology_range
    desc["filter"] = {
        "cache_placement": {"network_cache": cache_size},
        "workload": {"name": "STATIONARY", "alpha": alpha},
    }
    desc["ymetrics"] = [("CACHE_HIT_RATIO", "MEAN")] * len(strategies)
    desc["ycondnames"] = [("strategy", "name")] * len(strategies)
    desc["ycondvals"] = strategies
    desc["errorbar"] = True
    desc["legend_loc"] = "lower right"
    desc["bar_color"] = STRATEGY_BAR_COLOR
    desc["bar_hatch"] = STRATEGY_BAR_HATCH
    desc["legend"] = STRATEGY_LEGEND
    desc["plotempty"] = PLOT_EMPTY_GRAPHS
    plot_bar_chart(
        resultset,
        desc,
        "CACHE_HIT_RATIO_A={}_C={}.jpg".format(alpha, cache_size),
        plotdir,
    )

def plot_cache_hits_vs_alpha(
    resultset, topology, cache_size, alpha_range, strategies, plotdir
):
    if "NO_CACHE" in strategies:
        strategies.remove("NO_CACHE")
    desc = {}
    desc["title"] = "Cache hit ratio: T={} C={}".format(topology, cache_size)
    desc["ylabel"] = "Cache hit ratio"
    desc["xlabel"] = "Content distribution \u03b1"
    desc["xparam"] = ("workload", "alpha")
    desc["xvals"] = alpha_range
    desc["filter"] = {
        "topology": {"name": topology},
        "cache_placement": {"network_cache": cache_size},
    }
    desc["ymetrics"] = [("CACHE_HIT_RATIO", "MEAN")] * len(strategies)
    desc["ycondnames"] = [("strategy", "name")] * len(strategies)
    desc["ycondvals"] = strategies
    desc["errorbar"] = True
    desc["legend_loc"] = "upper left"
    desc["line_style"] = STRATEGY_STYLE
    desc["legend"] = STRATEGY_LEGEND
    desc["plotempty"] = PLOT_EMPTY_GRAPHS
    plot_lines(
        resultset,
        desc,
        "CACHE_HIT_RATIO_T={}@C={}.jpg".format(topology, cache_size),
        plotdir,
    )

def plot_latency_vs_alpha(
    resultset, topology, cache_size, alpha_range, strategies, plotdir
):
    desc = {}
    desc["title"] = "Latency: T={} C={}".format(topology, cache_size)
    desc["xlabel"] = "Content distribution \u03b1"
    desc["ylabel"] = "Latency (ms)"
    desc["xparam"] = ("workload", "alpha")
    desc["xvals"] = alpha_range
    desc["filter"] = {
        "topology": {"name": topology},
        "cache_placement": {"network_cache": cache_size},
    }
    desc["ymetrics"] = [("LATENCY", "MEAN")] * len(strategies)
    desc["ycondnames"] = [("strategy", "name")] * len(strategies)
    desc["ycondvals"] = strategies
    desc["errorbar"] = True
    desc["legend_loc"] = "upper right"
    desc["line_style"] = STRATEGY_STYLE
    desc["legend"] = STRATEGY_LEGEND
    desc["plotempty"] = PLOT_EMPTY_GRAPHS
    plot_lines(
        resultset, desc, "LATENCY_T={}@C={}.jpg".format(topology, cache_size), plotdir
    )

def plot_cost_vs_alpha(
    resultset, topology, cache_size, alpha_range, strategies, plotdir
):
    desc = {}
    desc["title"] = "Cost: T={} C={}".format(topology, cache_size)
    desc["xlabel"] = "Content distribution \u03b1"
    desc["ylabel"] = "Cost ($)"
    desc["xparam"] = ("workload", "alpha")
    desc["xvals"] = alpha_range
    desc["filter"] = {
        "topology": {"name": topology},
        "cache_placement": {"network_cache": cache_size},
    }
    desc["ymetrics"] = [("COST", "MEAN")] * len(strategies)
    desc["ycondnames"] = [("strategy", "name")] * len(strategies)
    desc["ycondvals"] = strategies
    desc["errorbar"] = True
    desc["legend_loc"] = "upper right"
    desc["line_style"] = STRATEGY_STYLE
    desc["legend"] = STRATEGY_LEGEND
    desc["plotempty"] = PLOT_EMPTY_GRAPHS
    plot_lines(
        resultset, desc, "COST_T={}@C={}.jpg".format(topology, cache_size), plotdir
    )

def plot_cost_components_alpha(
    resultset, topology, cache_size, alpha_range, strategies, plotdir
):
    """
    Plot cost components for each strategy as a stacked bar plot with strategy names under each bar.
    """
    # Cost component names in the result set
    cost_components = ["DEPRECIATION", "BANDWIDTH", "READ_STORAGE", "WRITE_STORAGE", "ROUTERS", "LINKS", "PENALTY"]
    num_components = len(cost_components)
    
    # Prepare for plotting
    fig, ax = plt.subplots(figsize=(10, 6))  # Increase figure size
    bar_width = 0.15  # Width of each strategy's bar
    bar_spacing = 0.05  # Extra space between groups of bars
    total_bars_per_group = len(strategies) * (bar_width + bar_spacing)
    
    # Generate positions for each bar, spacing them based on both cache sizes and strategies
    x_positions = []
    for i, alpha in enumerate(alpha_range):
        for j, strategy in enumerate(strategies):
            x_positions.append(i * (total_bars_per_group + 0.2) + j * (bar_width + bar_spacing))
    
    x_positions = np.array(x_positions)
    
    # Define color and hatch styles for each cost component
    cost_colors = ['#FF7F0E', '#1F77B4', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2']
    cost_hatches = ['/', '\\', '|', '-', '+', 'x', 'o']
    
    # Plot bars for each strategy and component
    for i, strategy in enumerate(strategies):
        bottom = np.zeros(len(alpha_range))  # Initialize for stacking bars

        for j, component in enumerate(cost_components):
            data = []
            for alpha in alpha_range:
                filtered = resultset.filter({
                    "topology": {"name": topology},
                    "cache_placement": {"network_cache": cache_size},
                    "strategy": {"name": strategy},
                    "workload" :{"name": "STATIONARY", "alpha": alpha},
                })
                
                cost = filtered[0][1]['COST'].get(component, 0) if len(filtered) > 0 else 0
                data.append(cost)
            
            ax.bar(
                x_positions[i::len(strategies)], data, bar_width,
                bottom=bottom,
                color=cost_colors[j],
                hatch=cost_hatches[j]
            )
            bottom += np.array(data)

    # Set labels, title, ticks, and legends
    ax.set_xlabel('Cache Proportion and Strategy', fontsize=14)
    ax.set_ylabel('Cost', fontsize=14)
    ax.set_title('Cost Components per Cache Size and Strategy', fontsize=16)
    
    # Add cache size and strategy labels as x-axis labels
    xtick_labels = []
    for alpha in alpha_range:
        for strategy in strategies:
            xtick_labels.append(f'{strategy}\n(Cache {alpha})')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xtick_labels, fontsize=10, rotation=45, ha="right")
    
    # Add gridlines
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Add a legend for the cost components only
    handles = [plt.Rectangle((0,0),1,1, color=cost_colors[i], hatch=cost_hatches[i]) for i in range(num_components)]
    ax.legend(handles, cost_components, loc='upper right', fontsize=10, title="Cost Components")

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir, f"COST_COMPONENTS_T={topology}@C={cache_size}.jpg"), bbox_inches='tight')
    # plt.show()cache_size_rangecache_size_range

def plot_chrcp_vs_alpha(
    resultset, topology, cache_size, alpha_range, strategies, plotdir):
    # Step 1: Filter results for the LCE strategy, specified topology, and CHRCP metric
    lce_filtered = resultset.filter({
        "topology": {"name": topology},
        "cache_placement": {"network_cache": cache_size},
        "strategy": {"name": "LCE"},
        "CHRCP": {}
    })
    
    # Step 2: Create a dictionary of LCE CHRCP values for normalization
    lce_cost = {
        res[0].get("cache_placement").get("network_cache"): res[1].get("COST").get("MEAN")
        for res in lce_filtered
        if res[1].get("COST").get("MEAN") is not None
    }
    
    if not lce_cost:
        logger.error("No LCE CHRCP values found for normalization.")
        return
    
    # Step 3: Normalize the resultset based on LCE CHRCP values
    for entry, metrics in resultset:
        alpha = entry.get("workload", {}).get("alpha")
        if alpha in lce_cost and metrics.get("CHRCP", {}).get("MEAN") is not None:
            normalized_value = metrics["CHRCP"]["MEAN"] / lce_cost[cache_size]
            metrics["CHRCP"]["MEAN"] = normalized_value
    
    # Step 4: Plot the normalized CHRCP results
    desc = {
        "title": f"CHRCP: T={topology} C={cache_size}",
        "xlabel": "Cache to population ratio",
        "ylabel": "CHRCP (relative to LCE)",
        "xscale": "log",
        "xparam": ("cache_placement", "network_cache"),
        "xvals": alpha_range,
        "filter": {
            "topology": {"name": topology},
            "cache_placement": {"network_cache": cache_size},
            "workload": {"name": "STATIONARY"},
        },
        "ymetrics": [("CHRCP", "MEAN")] * len(strategies),
        "ycondnames": [("strategy", "name")] * len(strategies),
        "ycondvals": strategies,
        "metric": ("CHRCP", "MEAN"),
        "errorbar": True,
        "legend_loc": "upper right",
        "line_style": STRATEGY_STYLE,
        "legend": STRATEGY_LEGEND,
        "plotempty": PLOT_EMPTY_GRAPHS,
    }
    
    plot_lines(
        resultset, desc, f"CHRCP_T={topology}@C={cache_size}.jpg", plotdir
    )


def run(config, results, plotdir):
    """Run the plot script

    Parameters
    ----------
    config : str
        The path of the configuration file
    results : str
        The file storing the experiment results
    plotdir : str
        The directory into which graphs will be saved
    """
    settings = Settings()
    settings.read_from(config)
    config_logging(settings.LOG_LEVEL)
    resultset = RESULTS_READER[settings.RESULTS_FORMAT](results)
    # Create dir if not existsing
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    # Parse params from settings
    topologies = settings.TOPOLOGIES
    cache_sizes = settings.NETWORK_CACHE
    strategies = settings.STRATEGIES
    alphas = settings.ALPHA
    # Plot graphs
    for topology in topologies:
        for cache_size in cache_sizes:
            logger.info(
                "Plotting cache hit ratio for topology %s and cache size %s vs alpha"
                % (topology, str(cache_size))
            )
            plot_cache_hits_vs_alpha(
                resultset, topology, cache_size, alphas, strategies, plotdir
            )
            logger.info(
                "Plotting latency for topology %s vs cache size %s"
                % (topology, str(cache_size))
            )
            plot_latency_vs_alpha(
                resultset, topology, cache_size, alphas, strategies, plotdir
            )
            logger.info(
                "Plotting cost for topology %s vs cache size %s"
                % (topology, str(cache_size))
            )
            plot_cost_vs_alpha(
                resultset, topology, cache_size, alphas, strategies, plotdir
            )
            logger.info(
                "Plotting cost components for topology %s vs cache size %s"
                % (topology, str(cache_size))
            )
            plot_cost_components_alpha(
            resultset, topology, cache_size, alphas, strategies, plotdir
            )
            logger.info(
                "Plotting chrcp for topology %s vs cache size %s"
                % (topology, str(cache_size))
            )
            plot_chrcp_vs_alpha(
            resultset, topology, cache_size, alphas, strategies, plotdir
            )

    for topology in topologies:
        for alpha in alphas:
            logger.info(
                "Plotting cache hit ratio for topology %s and alpha %s vs cache size"
                % (topology, str(alpha))
            )
            plot_cache_hits_vs_cache_size(
                resultset, topology, alpha, cache_sizes, strategies, plotdir
            )
            logger.info(
                "Plotting chrcp for topology %s vs cache size"
                % (topology)
            )
            plot_chrcp_vs_cache_size(
                resultset, topology, alpha, cache_sizes, strategies, plotdir
            )
            logger.info(
                "Plotting latency for topology %s and alpha %s vs cache size"
                % (topology, str(alpha))
            )
            plot_latency_vs_cache_size(
                resultset, topology, alpha, cache_sizes, strategies, plotdir
            )
            logger.info(
                "Plotting cost for topology %s vs cache size"
                % (topology)
            )
            plot_cost_vs_cache_size(
                resultset, topology, alpha, cache_sizes, strategies, plotdir
            )

            logger.info(
                "Plotting cost componenets %s vs cache size"
                % (topology)
            )
            plot_cost_components_vs_cache_size(
            resultset, topology, alpha, cache_sizes, strategies, plotdir
            )
            
    logger.info("Exit. Plots were saved in directory %s" % os.path.abspath(plotdir))


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
