#!/usr/bin/env python
"""Plot results read from a result set
"""
import os
import argparse
import logging

import matplotlib.pyplot as plt

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
    # "HR_SYMM": "b-o",
    # "HR_ASYMM": "g-D",
    # "HR_MULTICAST": "m-^",
    # "HR_HYBRID_AM": "c-s",
    # "HR_HYBRID_SM": "r-v",
    "LCE": "b--p",
     "Algo4": "m-^",
    # "LCD": "g-->",
    "CL4M": "g-->",
    "PROB_CACHE": "c--<",
    # "RAND_CHOICE": "r--<",
    # "RAND_BERNOULLI": "g--*",
    # "NO_CACHE": "k:o",
    # "OPTIMAL": "k-o",
    "COST_CACHE": "r-v",
}

# This dict maps name of strategies to names to be displayed in the legend
STRATEGY_LEGEND = {
    "LCE": "LCE",
    "Algo4": "Algo4",
    # "LCD": "LCD",
    # "HR_SYMM": "HR Symm",
    # "HR_ASYMM": "HR Asymm",
    # "HR_MULTICAST": "HR Multicast",
    # "HR_HYBRID_AM": "HR Hybrid AM",
    # "HR_HYBRID_SM": "HR Hybrid SM",
    "CL4M": "CacheLessForMore",
    "PROB_CACHE": "ProbCache",
    # "RAND_CHOICE": "Random (choice)",
    # "RAND_BERNOULLI": "Random (Bernoulli)",
    # "NO_CACHE": "No caching",
    # "OPTIMAL": "Optimal",
    "COST_CACHE": "CostCache",
}

# Color and hatch styles for bar charts of cache hit ratio and link load vs topology
STRATEGY_BAR_COLOR = {
    "LCE": "k",
    "Algo4" :"o.7",
    # "LCD": "0.4",
    # "NO_CACHE": "0.5",
    # "HR_ASYMM": "0.6",
    # "HR_SYMM": "0.7",
    "CL4M": "0.6",
    "PROB_CACHE": "0.5",
    "COST_CACHE": "0.4",
}

STRATEGY_BAR_HATCH = {
    "LCE": None,
    "Algo4" : "//",
    # "LCD": "//",
    # "NO_CACHE": "x",
    # "HR_ASYMM": "+",
    # "HR_SYMM": "\\",
    "CL4M": "x",
    "PROB_CACHE": "\\",
    "COST_CACHE": "+",
}


def plot_cache_hits_vs_cache_size(
    resultset, topology, cache_size_range, strategies, plotdir
):
    desc = {}
    if "NO_CACHE" in strategies:
        strategies.remove("NO_CACHE")
    desc["title"] = "Cache hit ratio: T={}".format(topology)
    desc["xlabel"] = "Cache to population ratio"
    desc["ylabel"] = "Cache hit ratio"
    desc["xscale"] = "log"
    desc["xparam"] = ("cache_placement", "network_cache")
    desc["xvals"] = cache_size_range
    desc["filter"] = {
        "topology": {"name": topology},
        "workload": {"name": "TRACE_DRIVEN"},
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
        "CACHE_HIT_RATIO_T={}.jpg".format(topology),
        plotdir,
    )

def plot_cost_vs_cache_size(
    resultset, topology, cache_size_range, strategies, plotdir
):
    desc = {}
    desc["title"] = "Cost: T={}".format(topology)
    desc["xlabel"] = "Cache to population ratio"
    desc["ylabel"] = "Cost"
    desc["xscale"] = "log"
    desc["xparam"] = ("cache_placement", "network_cache")
    desc["xvals"] = cache_size_range
    desc["filter"] = {
        "topology": {"name": topology},
        "workload": {"name": "TRACE_DRIVEN"},
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
        resultset, desc, "COST_T={}.jpg".format(topology), plotdir
    )

def plot_latency_vs_cache_size(
    resultset, topology, cache_size_range, strategies, plotdir
):
    desc = {}
    desc["title"] = "Latency: T={}".format(topology)
    desc["xlabel"] = "Cache to population ratio"
    desc["ylabel"] = "Latency"
    desc["xscale"] = "log"
    desc["xparam"] = ("cache_placement", "network_cache")
    desc["xvals"] = cache_size_range
    desc["filter"] = {
        "topology": {"name": topology},
        "workload": {"name": "TRACE_DRIVEN"},
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
        resultset, desc, "LATENCY_T={}.jpg".format(topology), plotdir
    )

def plot_cache_hits_vs_topology(
    resultset, cache_size, topology_range, strategies, plotdir
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
    desc["title"] = "Cache hit ratio: C={}".format(cache_size)
    desc["ylabel"] = "Cache hit ratio"
    desc["xparam"] = ("topology", "name")
    desc["xvals"] = topology_range
    desc["filter"] = {
        "cache_placement": {"network_cache": cache_size},
        "workload": {"name": "TRACE_DRIVEN"},
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
        "CACHE_HIT_RATIO_C={}.jpg".format(cache_size),
        plotdir,
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
    # topologies = settings.TOPOLOGIES
    cache_sizes = settings.NETWORK_CACHE
    strategies = settings.STRATEGIES
    # Plot graphs
    # for topology in topologies:
    topology = "PATH"
    logger.info(
        "Plotting cache hit ratio for topology %s vs cache size"
        % (topology)
    )
    plot_cache_hits_vs_cache_size(
        resultset, topology, cache_sizes, strategies, plotdir
    )
    logger.info(
        "Plotting latency for topology %s vs cache size"
        % (topology)
    )
    plot_latency_vs_cache_size(
        resultset, topology, cache_sizes, strategies, plotdir
    )
    logger.info(
        "Plotting cost for topology %s vs cache size"
        % (topology)
    )
    plot_cost_vs_cache_size(
        resultset, topology, cache_sizes, strategies, plotdir
    )
    # for cache_size in cache_sizes:
    #         logger.info(
    #             "Plotting cache hit ratio for cache size %s vs topologies"
    #             % (str(cache_size))
    #         )
    #         plot_cache_hits_vs_topology(
    #             resultset, cache_size, topologies, strategies, plotdir
    #         )
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
