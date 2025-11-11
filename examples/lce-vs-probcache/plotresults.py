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
    "CPCache":"ko-",
    "CL4M": "m->",
}

# This dict maps name of strategies to names to be displayed in the legend
STRATEGY_LEGEND = {
    "LCE": "LCE",
    "LCD": "LCD",
    "PROB_CACHE": "ProbCache",
    "RAND_CHOICE": "Random",
    "CL2SM": "CL2SM",
    "CPCache":"CPCache",
    "CL4M": "CL4M",
    # "HR_SYMM": "HR Symm",
    # "HR_ASYMM": "HR Asymm",
    # "HR_MULTICAST": "HR Multicast",
    # "HR_HYBRID_AM": "HR Hybrid AM",
    # "HR_HYBRID_SM": "HR Hybrid SM",
    # "RAND_BERNOULLI": "Random (Bernoulli)",
    # "NO_CACHE": "No caching",
    # "OPTIMAL": "Optimal",
}

PLACEMENT_STYLE = {
    "GREEN":"g->",
    "UNIFORM": "b-v",
    "DEGREE": "g-o",
    "BETWEENNESS_CENTRALITY" :"ko-", 
    "CONSOLIDATED":"k-d", 
    "RANDOM":"r-s", 
    "OPTIMAL_MEDIAN":"c-*", 
    "OPTIMAL_HASHROUTING":"m->",
    "HYBRID_GREEN_CENTRALITY":"m->",
    "ALLOCATED":"r-s", 
}

PLACEMENT_LEGEND = {
    "GREEN":"GREEN",
    "UNIFORM": "UNIFORM",
    "DEGREE": "DEGREE",
    "BETWEENNESS_CENTRALITY" :"BETWEENNESS_CENTRALITY", 
    "CONSOLIDATED":"CONSOLIDATED", 
    "RANDOM":"RANDOM", 
    "OPTIMAL_MEDIAN":"OPTIMAL_MEDIAN", 
    "OPTIMAL_HASHROUTING":"OPTIMAL_HASHROUTING",
    "HYBRID_GREEN_CENTRALITY":"HYBRID_GREEN_CENTRALITY",
    "ALLOCATED":"ALLOCATED",
}
# Color and hatch styles for bar charts of cache hit ratio and link load vs topology

STRATEGY_BAR_COLOR = {
    strategy: MATPLOTLIB_COLOR_TO_HEX[style[0]]
    for strategy, style in STRATEGY_STYLE.items()
    if style[0] in MATPLOTLIB_COLOR_TO_HEX
}

STRATEGY_BAR_HATCH = {
    "CL2SM": None,
    "LCE": "\\",
    "LCD": "//",
    "PROB_CACHE": "x",
    "CL4M": "+",
    "CPCache": "."
}


def plot_cache_hits_vs_cache_size(
    resultset, topology, alpha, cache_size_range, strategies, plotdir
):
    desc = {}
    if "NO_CACHE" in strategies:
        strategies.remove("NO_CACHE")
    # print("here")
    desc["xlabel"] = "Cache Proportion (%)"
    desc["ylabel"] = "Cache hit ratio"
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
    # desc["title"] = "Content Popularity \u03b1 = {}".format(alpha)
    desc["xlabel"] = "Cache Proportion (%)"
    desc["ylabel"] = "Cost per Request ($)"
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
    # lce_filtered = resultset.filter({
    #     "topology": {"name": topology},
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
    #     "topology": {"name": topology},
    #     "workload" :{"alpha":alpha},
    #     "CHRCP": {}
    # })

    # if not lce_cost:
    #     logger.error("No LCE CHRCP values found for normalization.")
    #     return
    # # Step 3: Normalize the resultset based on LCE CHRCP values
    # for entry, metrics in alpha_filtered:
    #     cache_size = entry.get("cache_placement", {}).get("network_cache")
    #     if cache_size in lce_cost and metrics.get("CHRCP", {}).get("MEAN") is not None:
    #         normalized_value = metrics["CHRCP"]["MEAN"] / lce_cost[cache_size]
    #         metrics["CHRCP"]["MEAN"] = normalized_value
    
    # Step 4: Plot the normalized CHRCP results
    desc = {
        # "title": f"Content Popularity \u03b1 = {alpha}",
        "xlabel": "Cache Proportion (%)",
        "ylabel": "CHRCP (relative to LCE)",
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
    # desc["title"] = "Content Popularity \u03b1 = {}".format(alpha)
    desc["xlabel"] = "Cache Proportion (%)"
    desc["ylabel"] = "Latency (ms)"
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
    # ax.set_title('Cost Components per Cache Size and Strategy', fontsize=16)
    
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

def plot_network_cf_components_vs_cache_size(resultset, topology, alpha, cache_size_range, placements, plotdir):
    """
    Plot ROUTERS_OPEX and LINKS_OPEX for each placement and cache size
    as stacked bars showing network carbon footprint components.
    """
    # Components from CARBONFOOTPRINT
    cf_components = ["ROUTERS_OPEX", "LINKS_OPEX"]
    num_components = len(cf_components)

    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.18
    group_spacing = 0.4  # spacing between cache size groups

    cf_colors = ['#1F77B4', '#FF7F0E']  # blue = routers, orange = links
    cf_hatches = ['/', '\\']

    # --- Prepare data structures ---
    total_groups = len(cache_size_range)
    total_placements = len(placements)
    total_bars_per_group = total_placements

    all_data = {comp: [] for comp in cf_components}
    x_positions, xtick_labels = [], []

    # --- Collect data ---
    for i, cache_size in enumerate(cache_size_range):
        for k, placement in enumerate(placements):
            filtered = resultset.filter({
                "topology": {"name": topology},
                "cache_placement": {"network_cache": cache_size, "name": placement},
                "workload": {"name": "STATIONARY", "alpha": alpha},
            })

            cf_data = filtered[0][1]['CARBONFOOTPRINT'] if filtered else {}

            for comp in cf_components:
                val = cf_data.get(comp, 0)
                all_data[comp].append(val)

            x_pos = i * (total_placements * (bar_width + group_spacing)) + k * (bar_width + 0.1)
            x_positions.append(x_pos)
            xtick_labels.append(f"{placement}\n(Cache {cache_size})")

    x_positions = np.array(x_positions)
    bottom = np.zeros(len(x_positions))

    # --- Plot stacked bars ---
    for j, comp in enumerate(cf_components):
        data = np.array(all_data[comp])
        ax.bar(
            x_positions,
            data,
            bar_width,
            bottom=bottom,
            color=cf_colors[j],
            hatch=cf_hatches[j],
            label=comp,
            edgecolor="black"
        )
        bottom += data

    # --- Style and labels ---
    ax.set_xlabel("Cache Proportion and Placement", fontsize=13)
    ax.set_ylabel("Network Carbon Footprint (gCO₂e)", fontsize=13)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(title="Network CF Components", fontsize=10)
    plt.tight_layout()

    # --- Save ---
    outfile = os.path.join(plotdir, f"ROUTERS_vs_LINKS_OPEX_T={topology}@A={alpha}.jpg")
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
    print(f"[✔] Saved network CF plot to {outfile}")

def plot_opex_vs_capex_cache_size(resultset, topology, alpha, cache_size_range, placements, plotdir):
    """
    Plot stacked OPEX + CAPEX carbon footprint for each placement and cache size.
    """
    cf_components = ["TOTAL_OPEX", "TOTAL_CAPEX"]
    num_components = len(cf_components)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.15
    bar_spacing = 0.05
    total_bars_per_group = len(placements) * (bar_width + bar_spacing)

    cf_colors = ['#1F77B4', '#FF7F0E']  # Blue for OPEX, Orange for CAPEX
    cf_hatches = ['/', '\\']

    x_positions, xtick_labels = [], []

    # Collect all bar data
    all_data = {comp: [] for comp in cf_components}

    for i, cache_size in enumerate(cache_size_range):
        for k, placement in enumerate(placements):
            filtered = resultset.filter({
                "topology": {"name": topology},
                "cache_placement": {"network_cache": cache_size, "name": placement},
                "workload": {"name": "STATIONARY", "alpha": alpha},
            })

            cf_data = filtered[0][1]['CARBONFOOTPRINT'] if filtered else {}
            for comp in cf_components:
                value = cf_data.get(comp, 0)
                all_data[comp].append(value)

            x_pos = i * (total_bars_per_group + 0.2) + k * (bar_width + bar_spacing)
            x_positions.append(x_pos)
            xtick_labels.append(f"{placement}\n(Cache {cache_size})")

    x_positions = np.array(x_positions)
    bottom = np.zeros(len(x_positions))

    # Draw stacked bars
    for j, comp in enumerate(cf_components):
        data = np.array(all_data[comp])
        ax.bar(
            x_positions, data, bar_width,
            bottom=bottom,
            color=cf_colors[j],
            hatch=cf_hatches[j],
            label=comp,
            edgecolor='black'
        )
        bottom += data

    # Labels and layout
    ax.set_xlabel('Cache Proportion and Placement', fontsize=13)
    ax.set_ylabel('Carbon Footprint (gCO₂e)', fontsize=13)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xtick_labels, fontsize=10, rotation=45, ha="right")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(title="CF Component", fontsize=10)
    plt.tight_layout()

    plt.savefig(os.path.join(plotdir, f"CAPEX_OPEX_T={topology}@A={alpha}.jpg"), bbox_inches='tight')
    plt.close()

def plot_server_opex_vs_capex_cache_size(resultset, topology, alpha, cache_size_range, placements, plotdir):
    """
    Plot SERVER_OPEX vs SERVER_CAPEX for each placement and cache size
    as grouped side-by-side bars.
    """
    # Define cost components
    opex_capex = ["SERVER_OPEX", "SERVER_CAPEX"]
    num_components = len(opex_capex)
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(11, 6))
    bar_width = 0.18
    group_spacing = 0.4  # spacing between cache size groups

    cf_colors = ['#1F77B4', '#FF7F0E']  # blue = OPEX, orange = CAPEX
    cf_hatches = ['/', '\\']

    # -----------------------------
    # Compute X positions
    # -----------------------------
    total_groups = len(cache_size_range)
    total_placements = len(placements)
    total_bars_per_group = total_placements * num_components

    x_positions, xtick_labels = [], []
    all_data = {component: [] for component in opex_capex}

    # -----------------------------
    # Gather data
    # -----------------------------
    for i, cache_size in enumerate(cache_size_range):
        for k, placement in enumerate(placements):
            filtered = resultset.filter({
                "topology": {"name": topology},
                "cache_placement": {"network_cache": cache_size, "name": placement},
                "workload": {"name": "STATIONARY", "alpha": alpha},
            })

            cf_data = filtered[0][1]['CARBONFOOTPRINT'] if filtered else {}
            for comp in opex_capex:
                val = cf_data.get(comp, 0)
                all_data[comp].append(val)

            x_positions.append(i * (total_placements * (num_components * bar_width + group_spacing)) +
                               k * (num_components * bar_width + 0.1))
            xtick_labels.append(f'{placement}\n(Cache {cache_size})')

    x_positions = np.array(x_positions)

    # -----------------------------
    # Plot side-by-side bars
    # -----------------------------
    for j, component in enumerate(opex_capex):
        data = np.array(all_data[component])
        offset = j * (bar_width + 0.02)
        ax.bar(
            x_positions + offset,
            data,
            bar_width,
            label=component,
            color=cf_colors[j],
            hatch=cf_hatches[j],
            edgecolor='black'
        )

    # -----------------------------
    # Axis and layout
    # -----------------------------
    ax.set_xlabel('Cache Proportion and Placement', fontsize=13)
    ax.set_ylabel('Carbon Footprint (gCO₂e)', fontsize=13)
    ax.set_xticks(x_positions + bar_width / 2)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=10)
    ax.legend(title="Server CF Component", fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    outfile = os.path.join(plotdir, f"SERVER_CAPEX_VS_OPEX_T={topology}@A={alpha}.jpg")
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()
    print(f"[✔] Saved plot to {outfile}")

def plot_cf_vs_cache_size(
    resultset, topology, alpha, cache_size_range, cache_placements, plotdir
):
    desc = {}
    desc["xlabel"] = "Cache Proportion (%)"
    desc["ylabel"] = "Carbon Footprint g.CO2"
    desc["xparam"] = ("cache_placement", "network_cache")
    desc["xvals"] = cache_size_range
    desc["filter"] = {
        "topology": {"name": topology},
        "workload": {"name": "STATIONARY", "alpha": alpha},
    }
    desc["ymetrics"] = [("CARBONFOOTPRINT", "TOTAL")] * len(cache_placements)
    desc["ycondnames"] = [("cache_placement", "name")] * len(cache_placements)
    desc["ycondvals"] = cache_placements
    desc["metric"] = ("CARBONFOOTPRINT", "TOTAL")
    desc["errorbar"] = True
    desc["legend_loc"] = "upper left"
    desc["line_style"] = PLACEMENT_STYLE
    desc["legend"] = PLACEMENT_LEGEND
    desc["plotempty"] = PLOT_EMPTY_GRAPHS
    plot_lines(
        resultset,
        desc,
        "TOTAL_OPEX_CAPEX_T={}@A={}.jpg".format(topology, alpha),
        plotdir,
    )

def plot_capex_vs_cache_size(
    resultset, topology, alpha, cache_size_range, cache_placements, plotdir
):
    desc = {}
    desc["xlabel"] = "Cache Proportion (%)"
    desc["ylabel"] = "CAPEX g.CO2"
    desc["xparam"] = ("cache_placement", "network_cache")
    desc["xvals"] = cache_size_range
    desc["filter"] = {
        "topology": {"name": topology},
        "workload": {"name": "STATIONARY", "alpha": alpha},
    }
    desc["ymetrics"] = [("CARBONFOOTPRINT", "TOTAL_CAPEX")] * len(cache_placements)
    desc["ycondnames"] = [("cache_placement", "name")] * len(cache_placements)
    desc["ycondvals"] = cache_placements
    desc["metric"] = ("CARBONFOOTPRINT", "TOTAL_CAPEX")
    desc["errorbar"] = True
    desc["legend_loc"] = "upper left"
    desc["line_style"] = PLACEMENT_STYLE
    desc["legend"] = PLACEMENT_LEGEND
    desc["plotempty"] = PLOT_EMPTY_GRAPHS
    plot_lines(
        resultset,
        desc,
        "CAPEX_T={}@A={}.jpg".format(topology, alpha),
        plotdir,
    )

def plot_opex_vs_cache_size(
    resultset, topology, alpha, cache_size_range, placements, plotdir
):
    desc = {}
    desc["xlabel"] = "Cache Proportion (%)"
    desc["ylabel"] = "OPEX g.CO2"
    desc["xparam"] = ("cache_placement", "network_cache")
    desc["xvals"] = cache_size_range
    desc["filter"] = {
        "topology": {"name": topology},
        "workload": {"name": "STATIONARY", "alpha": alpha},
    }
    desc["ymetrics"] = [("CARBONFOOTPRINT", "TOTAL_OPEX")] * len(placements)
    desc["ycondnames"] = [("cache_placement", "name")] * len(placements)
    desc["ycondvals"] = placements
    desc["errorbar"] = True
    desc["legend_loc"] = "upper left"
    desc["line_style"] = PLACEMENT_STYLE
    desc["legend"] = PLACEMENT_LEGEND
    desc["plotempty"] = PLOT_EMPTY_GRAPHS
    plot_lines(
        resultset,
        desc,
        "OPEX_T={}@A={}.jpg".format(topology, alpha),
        plotdir,
    )

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
    desc["xlabel"] = "Topologies"
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

def plot_cost_vs_topology(
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
    desc["xlabel"] = "Topologies"
    desc["ylabel"] = "Cost per Request ($)"
    desc["xparam"] = ("topology", "name")
    desc["xvals"] = topology_range
    desc["filter"] = {
        "cache_placement": {"network_cache": cache_size},
        "workload": {"name": "STATIONARY", "alpha": alpha},
    }
    desc["ymetrics"] = [("COST", "MEAN")] * len(strategies)
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
        "COST_A={}_C={}.jpg".format(alpha, cache_size),
        plotdir,
    )

def plot_latency_vs_topology(
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
    desc["xlabel"] = "Topologies"
    desc["ylabel"] = "Latency (ms)"
    desc["xparam"] = ("topology", "name")
    desc["xvals"] = topology_range
    desc["filter"] = {
        "cache_placement": {"network_cache": cache_size},
        "workload": {"name": "STATIONARY", "alpha": alpha},
    }
    desc["ymetrics"] = [("LATENCY", "MEAN")] * len(strategies)
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
        "LATENCY_A={}_C={}.jpg".format(alpha, cache_size),
        plotdir,
    )

def plot_chrcp_vs_topology(
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
    desc["xlabel"] = "Topologies"
    desc["ylabel"] = "CHRCP (relative to LCE)"
    desc["xparam"] = ("topology", "name")
    desc["xvals"] = topology_range
    desc["filter"] = {
        "cache_placement": {"network_cache": cache_size},
        "workload": {"name": "STATIONARY", "alpha": alpha},
    }
    desc["ymetrics"] = [("CHRCP", "MEAN")] * len(strategies)
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
        "CHRCP_A={}_C={}.jpg".format(alpha, cache_size),
        plotdir,
    )

def plot_cache_hits_vs_alpha(
    resultset, topology, cache_size, alpha_range, strategies, plotdir
):
    if "NO_CACHE" in strategies:
        strategies.remove("NO_CACHE")
    desc = {}
    # desc["title"] = "Cache size = {}".format(cache_size)
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
    # desc["title"] = "Cache size = {}".format(cache_size)
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
    desc["xlabel"] = "Content distribution \u03b1"
    desc["ylabel"] = "Cost per Request ($)"
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

def plot_chrcp_vs_alpha(
    resultset, topology, cache_size, alpha_range, strategies, plotdir):
    # Step 1: Filter results for the LCE strategy, specified topology, and CHRCP metric
    # lce_filtered = resultset.filter({
    #     "topology": {"name": topology},
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
    #     "topology": {"name": topology},
    #     "workload" :{"alpha":alpha},
    #     "CHRCP": {}
    # })

    # if not lce_cost:
    #     logger.error("No LCE CHRCP values found for normalization.")
    #     return
    # # Step 3: Normalize the resultset based on LCE CHRCP values
    # for entry, metrics in alpha_filtered:
    #     cache_size = entry.get("cache_placement", {}).get("network_cache")
    #     if cache_size in lce_cost and metrics.get("CHRCP", {}).get("MEAN") is not None:
    #         normalized_value = metrics["CHRCP"]["MEAN"] / lce_cost[cache_size]
    #         metrics["CHRCP"]["MEAN"] = normalized_value
    
    # Step 4: Plot the normalized CHRCP results
    desc = {
        # "title": f"Content Popularity \u03b1 = {alpha}",
        "xlabel": "Content distribution \u03b1",
        "ylabel": "CHRCP (relative to LCE)",
        "xparam": ("workload", "alpha"),
        "xvals": alpha_range,
        "filter": {
            "topology": {"name": topology},
            "cache_placement": {"network_cache": cache_size},
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

def plot_num_contents_vs_replicas(distributions, plotdir, filename="replica_distribution.png"):
    import pandas as pd

    # Get full replica range across all strategies
    all_replicas = sorted({r for d in distributions.values() for r in d})

    # Fill in missing replicas with count = 0
    records = []
    for strategy in distributions:
        for replica in all_replicas:
            count = distributions[strategy].get(replica, 0)
            records.append({
                "strategy": {"name": strategy},
                "replica": replica,
                "num_contents": count
            })

    # Mock resultset to work with plot_lines
    class MockVal:
        def __init__(self, d): self.d = d
        def getval(self, key):
            if isinstance(key, tuple):
                d = self.d
                for k in key:
                    if not isinstance(d, dict):
                        return None
                    d = d.get(k)
                return d
            return self.d.get(key)

    class MockResultSet:
        def __init__(self, records): self.records = records
        def filter(self, tree):
            # Pull out known values
            replica_val = tree.getval("replica")
            strat_val = tree.getval(("strategy", "name"))

            def match(r):
                if replica_val is not None and r.get("replica") != replica_val:
                    return False
                if strat_val is not None and r.get("strategy", {}).get("name") != strat_val:
                    return False
                return True

            return [(None, MockVal(r)) for r in self.records if match(r)]

    resultset = MockResultSet(records)

    strategies = list(distributions.keys())
    # replica_range = sorted({r for d in distributions.values() for r in d})

    desc = {
        "xlabel": "Number of Replicas",
        "ylabel": "Number of Contents",
        "xparam": "replica",
        "xvals": all_replicas,
        "ymetrics": ["num_contents"] * len(strategies),  # ← flat key now
        "ycondnames": [("strategy", "name")] * len(strategies),
        "ycondvals": strategies,
        "legend": STRATEGY_LEGEND,
        "line_style": STRATEGY_STYLE,
        "plotempty": PLOT_EMPTY_GRAPHS,
        "errorbar" : False,
        "legend_loc": "best",
        "yticks": list(range(0, 5001, 1000)),
    }

    plot_lines(resultset, desc, filename, plotdir)

def plot_cache_hits_vs_cache_placement(
    resultset, topology, alpha, cache_size_range, strategy, cache_placements, plotdir
):
    desc = {}
    desc["xlabel"] = "Cache Placements"
    desc["ylabel"] = "Cache hit ratio"
    desc["xparam"] = ("cache_placement", "network_cache")
    desc["xvals"] = cache_size_range
    desc["filter"] = {
        "topology": {"name": topology},
        "workload": {"name": "STATIONARY", "alpha": alpha},
        "strategy": {"name": strategy},
    }
    desc["ymetrics"] = [("CACHE_HIT_RATIO", "MEAN")] * len(cache_placements)
    desc["ycondnames"] = [("cache_placement", "name")] * len(cache_placements)
    desc["ycondvals"] = cache_placements
    desc["errorbar"] = True
    desc["legend_loc"] = "upper left"
    desc["line_style"] = PLACEMENT_STYLE
    desc["legend"] = PLACEMENT_LEGEND
    desc["plotempty"] = PLOT_EMPTY_GRAPHS
    print(desc)
    plot_lines(
        resultset,
        desc,
        "CACHE_HIT_RATIO_T={}@A={}@S={}.jpg".format(topology, alpha, strategy),
        plotdir,
    )

def plot_cost_vs_cache_placement(
    resultset, topology, alpha, cache_size_range, strategy, cache_placements, plotdir
):
    """
    Plot average cost per request ($) across different cache placements and cache sizes.
    Produces a multi-line plot (one line per cache placement) showing cost trends
    as cache size increases.

    Parameters
    ----------
    resultset : ResultSet
        The result set containing simulation results.
    topology : str
        Name of the topology (e.g., "GARR", "GEANT").
    alpha : float
        Zipf alpha parameter.
    cache_size_range : list[float]
        List of network cache proportions (e.g., [0.01, 0.015, 0.02]).
    strategy : str
        Name of the caching strategy (e.g., "CL2SM").
    cache_placements : list[str]
        List of cache placement strategies (e.g., ["ALLOCATED", "GREEN", "UNIFORM"]).
    plotdir : str
        Directory to save the output plot.
    """
    # --- Plot metadata description for the plotting engine ---
    desc = {
        "xlabel": "Cache Placement",
        "ylabel": "Average Cost per Request ($)",
        "xparam": ("cache_placement", "network_cache"),
        "xvals": cache_size_range,
        "filter": {
            "topology": {"name": topology},
            "workload": {"name": "STATIONARY", "alpha": alpha},
            "strategy": {"name": strategy},
        },
        "ymetrics": [("COST", "MEAN")] * len(cache_placements),
        "ycondnames": [("cache_placement", "name")] * len(cache_placements),
        "ycondvals": cache_placements,
        "metric": ("COST", "MEAN"),
        "errorbar": True,
        "legend_loc": "upper right",
        "line_style": PLACEMENT_STYLE,
        "legend": PLACEMENT_LEGEND,
        "plotempty": PLOT_EMPTY_GRAPHS,
    }

    # --- Generate and save the plot ---
    outfile = f"COST_T={topology}@A={alpha}@S={strategy}.jpg"
    plot_lines(resultset, desc, outfile, plotdir)
    print(f"[✔] Saved cost-per-request plot to {os.path.join(plotdir, outfile)}")

def plot_chrcp_vs_cache_placement(
    resultset, topology, alpha, cache_size_range, strategy, cache_placements, plotdir):
    # Step 1: Filter results for the LCE strategy, specified topology, and CHRCP metric
    # lce_filtered = resultset.filter({
    #     "topology": {"name": topology},
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
    #     "topology": {"name": topology},
    #     "workload" :{"alpha":alpha},
    #     "CHRCP": {}
    # })

    # if not lce_cost:
    #     logger.error("No LCE CHRCP values found for normalization.")
    #     return
    # # Step 3: Normalize the resultset based on LCE CHRCP values
    # for entry, metrics in alpha_filtered:
    #     cache_size = entry.get("cache_placement", {}).get("network_cache")
    #     if cache_size in lce_cost and metrics.get("CHRCP", {}).get("MEAN") is not None:
    #         normalized_value = metrics["CHRCP"]["MEAN"] / lce_cost[cache_size]
    #         metrics["CHRCP"]["MEAN"] = normalized_value
    
    # Step 4: Plot the normalized CHRCP results
    desc = {
        # "title": f"Content Popularity \u03b1 = {alpha}",
        "xlabel": "Cache Proportion (%)",
        "ylabel": "CHRCP (relative to LCE)",
        "xparam": ("cache_placement", "network_cache"),
        "xvals": cache_size_range,
        "filter": {
            "topology": {"name": topology},
            "workload": {"name": "STATIONARY", "alpha": alpha},
            "strategy": {"name": strategy},
        },
        "ymetrics": [("CHRCP", "MEAN")] * len(cache_placements),
        "ycondnames": [("cache_placement", "name")] * len(cache_placements),
        "ycondvals": cache_placements,
        "metric": ("CHRCP", "MEAN"),
        "errorbar": True,
        "legend_loc": "upper right",
        "line_style": PLACEMENT_STYLE,
        "legend": PLACEMENT_LEGEND,
        "plotempty": PLOT_EMPTY_GRAPHS,
    }
    
    plot_lines(
        resultset, desc, f"CHRCP_T={topology}@A={alpha}@S={strategy}.jpg", plotdir
    )

def plot_latency_vs_cache_placement(
    resultset, topology, alpha, cache_size_range, strategy, cache_placements, plotdir
):
    desc = {}
    # desc["title"] = "Content Popularity \u03b1 = {}".format(alpha)
    desc["xlabel"] = "Cache Placements"
    desc["ylabel"] = "Latency (ms)"
    desc["xparam"] = ("cache_placement", "network_cache")
    desc["xvals"] = cache_size_range 
    desc["filter"] = {
        "topology": {"name": topology},
        "workload": {"name": "STATIONARY", "alpha": alpha},
        "strategy": {"name": strategy},
    }
    desc["ymetrics"] = [("LATENCY", "MEAN")] * len(cache_placements)
    desc["ycondnames"] = [("cache_placement", "name")] * len(cache_placements)
    desc["ycondvals"] = cache_placements
    desc["metric"] = ("LATENCY", "MEAN")
    desc["errorbar"] = True
    desc["legend_loc"] = "upper right"
    desc["line_style"] = PLACEMENT_STYLE
    desc["legend"] = PLACEMENT_LEGEND
    desc["plotempty"] = PLOT_EMPTY_GRAPHS
    print(desc)
    plot_lines(
        resultset, desc, "LATENCY_T={}@A={}@S={}.jpg".format(topology, alpha, strategy), plotdir
    )

def plot_cf_vs_cache_placement(
    resultset, topology, alpha, cache_size_range, strategy, cache_placements, plotdir
):
    """
    Plot total carbon footprint (OPEX + CAPEX) versus cache placement and cache size.

    Parameters
    ----------
    resultset : ResultSet
        The simulation results container.
    topology : str
        Network topology name.
    alpha : float
        Workload alpha parameter.
    cache_size_range : list
        List of cache proportions (e.g., [0.01, 0.02, 0.05]).
    strategy : str
        Caching or management strategy name.
    cache_placements : list
        List of cache placement names (e.g., ["UNIFORM", "ALLOCATED"]).
    plotdir : str
        Output directory for saving plots.
    """

    desc = {
        "xlabel": "Cache Placement",
        "ylabel": "Total Carbon Footprint (gCO₂e)",
        "xparam": ("cache_placement", "network_cache"),
        "xvals": cache_size_range,
        "filter": {
            "topology": {"name": topology},
            "workload": {"name": "STATIONARY", "alpha": alpha},
            "strategy": {"name": strategy},
        },
        "ymetrics": [("CARBONFOOTPRINT", "TOTAL")] * len(cache_placements),
        "ycondnames": [("cache_placement", "name")] * len(cache_placements),
        "ycondvals": cache_placements,
        "errorbar": True,
        "legend_loc": "upper left",
        "line_style": PLACEMENT_STYLE,
        "legend": PLACEMENT_LEGEND,
        "plotempty": PLOT_EMPTY_GRAPHS,
    }

    filename = f"TOTAL_CF_T={topology}@A={alpha}@S={strategy}.jpg"
    plot_lines(resultset, desc, filename, plotdir)
    print(f"[✔] Saved total CF plot → {os.path.join(plotdir, filename)}")

def plot_capex_vs_cache_placement(
    resultset, topology, alpha, cache_size_range, strategy, cache_placements, plotdir
):
    """
    Plot embodied carbon footprint (CAPEX) versus cache placement and cache size.
    Produces a multi-line plot (one per cache placement) showing TOTAL_CAPEX
    from CARBONFOOTPRINT results.
    """
    desc = {
        "xlabel": "Cache Placement",
        "ylabel": "CAPEX (gCO₂e)",
        "xparam": ("cache_placement", "network_cache"),
        "xvals": cache_size_range,
        "filter": {
            "topology": {"name": topology},
            "workload": {"name": "STATIONARY", "alpha": alpha},
            "strategy": {"name": strategy},
        },
        "ymetrics": [("CARBONFOOTPRINT", "TOTAL_CAPEX")] * len(cache_placements),
        "ycondnames": [("cache_placement", "name")] * len(cache_placements),
        "ycondvals": cache_placements,
        "errorbar": True,
        "legend_loc": "upper left",
        "line_style": PLACEMENT_STYLE,
        "legend": PLACEMENT_LEGEND,
        "plotempty": PLOT_EMPTY_GRAPHS,
    }

    outfile = f"CAPEX_T={topology}@A={alpha}@S={strategy}.jpg"
    plot_lines(resultset, desc, outfile, plotdir)
    print(f"[✔] Saved CAPEX plot to {os.path.join(plotdir, outfile)}")

def plot_opex_vs_cache_placement(
    resultset, topology, alpha, cache_size_range, strategy, cache_placements, plotdir
):
    """
    Plot operational carbon footprint (OPEX) versus cache placement and cache size.
    Produces a multi-line plot (one per cache placement) showing TOTAL_OPEX
    from CARBONFOOTPRINT results.
    """
    desc = {
        "xlabel": "Cache Placement",
        "ylabel": "OPEX (gCO₂e)",
        "xparam": ("cache_placement", "network_cache"),
        "xvals": cache_size_range,
        "filter": {
            "topology": {"name": topology},
            "workload": {"name": "STATIONARY", "alpha": alpha},
            "strategy": {"name": strategy},
        },
        "ymetrics": [("CARBONFOOTPRINT", "TOTAL_OPEX")] * len(cache_placements),
        "ycondnames": [("cache_placement", "name")] * len(cache_placements),
        "ycondvals": cache_placements,
        "errorbar": True,
        "legend_loc": "upper left",
        "line_style": PLACEMENT_STYLE,
        "legend": PLACEMENT_LEGEND,
        "plotempty": PLOT_EMPTY_GRAPHS,
    }

    outfile = f"OPEX_T={topology}@A={alpha}@S={strategy}.jpg"
    plot_lines(resultset, desc, outfile, plotdir)
    print(f"[✔] Saved OPEX plot to {os.path.join(plotdir, outfile)}")

def plot_link_load_vs_cache_placement(
    resultset, topology, alpha, cache_size_range, strategy, cache_placements, plotdir
):
    desc = {}
    desc["xlabel"] = "Cache Placements"
    desc["ylabel"] = "Internal Link Load"
    desc["xparam"] = ("cache_placement", "network_cache")
    desc["xvals"] = cache_size_range
    desc["filter"] = {
        "topology": {"name": topology},
        "workload": {"name": "STATIONARY", "alpha": alpha},
        "strategy": {"name": strategy},
    }
    desc["ymetrics"] = [("LINK_LOAD", "MEAN_INTERNAL")] * len(cache_placements)
    desc["ycondnames"] = [("cache_placement", "name")] * len(cache_placements)
    desc["ycondvals"] = cache_placements
    desc["errorbar"] = True
    desc["legend_loc"] = "upper left"
    desc["line_style"] = PLACEMENT_STYLE
    desc["legend"] = PLACEMENT_LEGEND
    desc["plotempty"] = PLOT_EMPTY_GRAPHS
    print(desc)
    plot_lines(
        resultset,
        desc,
        "LINK_LOAD_INTERNAL_T={}@A={}@S={}.jpg".format(topology, alpha, strategy),
        plotdir,
    )

# def plot_per_tier_opex_vs_capex_all_configs(
#     resultset,
#     topology,
#     alpha,
#     cache_size_range,
#     placements,
#     plotdir,
#     mode='global'  # 'global' or 'per-tier'
# ):
#     """
#     Generates two plots:
#     (1) Fix cache size -> vary placement
#     (2) Fix placement -> vary cache size
#     Each bar = tier (e.g., DRAM/SSD) for a given configuration.

#     Automatically reads TIER_STATS from resultset for each simulation.
#     """
#     import os
#     import numpy as np
#     import matplotlib.pyplot as plt

#     cf_colors = ['#1F77B4', '#FF7F0E']  # OPEX, CAPEX
#     cf_labels = ['OPEX Fraction', 'CAPEX Fraction']
#     bar_width = 0.35
#     group_spacing = 0.25

#     def compute_fractions(per_tier, tier_stats):
#         """Compute OPEX/CAPEX fractions based on mode and tier stats."""
#         # Ensure we have valid tier stats
#         if not tier_stats:
#             tier_stats = {t: 1 for t in per_tier.keys()}

#         # --- Scale each tier by its node count
#         scaled_cf = {
#             t: (v.get("OPEX", 0) + v.get("CAPEX", 0)) * tier_stats.get(t, 1)
#             for t, v in per_tier.items()
#         }

#         total_cf = sum(scaled_cf.values())

#         fractions = {}
#         for t, v in per_tier.items():
#             opex = v.get("OPEX", 0) * tier_stats.get(t, 1)
#             capex = v.get("CAPEX", 0) * tier_stats.get(t, 1)
#             if mode == 'global' and total_cf > 0:
#                 opex_frac = opex / total_cf
#                 capex_frac = capex / total_cf
#             elif mode == 'per-tier':
#                 tier_total = opex + capex
#                 opex_frac = opex / tier_total if tier_total > 0 else 0
#                 capex_frac = capex / tier_total if tier_total > 0 else 0
#             else:
#                 opex_frac = capex_frac = 0
#             fractions[t] = (opex_frac, capex_frac)
#         return fractions

#     # ---------- (A) FIX CACHE SIZE ----------
#     for cache_size in cache_size_range:
#         fig, ax = plt.subplots(figsize=(10, 5))
#         x_positions, xtick_labels = [], []
#         offset = 0

#         for placement in placements:
#             filtered = resultset.filter({
#                 "topology": {"name": topology},
#                 "cache_placement": {"network_cache": cache_size},
#                 "cache_placement": {"name": placement},
#                 "workload": {"name": "STATIONARY", "alpha": alpha},
#             })
#             if len(filtered) == 0:
#                 continue

#             result = filtered[0][1]
#             cf_data = result.get("CARBONFOOTPRINT", {})
#             per_tier = cf_data.get("PER_TIER", {})
#             tier_stats = cf_data.get("TIER_STATS", result.get("TIER_STATS", {}))

#             if not per_tier:
#                 continue

#             fractions = compute_fractions(per_tier, tier_stats)

#             for tier_name, (opex_frac, capex_frac) in fractions.items():
#                 ax.bar(offset, opex_frac, bar_width, color=cf_colors[0],
#                        hatch='/', edgecolor='black', label=cf_labels[0] if offset == 0 else "")
#                 ax.bar(offset, capex_frac, bar_width, bottom=opex_frac,
#                        color=cf_colors[1], hatch='\\', edgecolor='black', label=cf_labels[1] if offset == 0 else "")

#                 xtick_labels.append(f"{tier_name}\n{placement}")
#                 x_positions.append(offset)
#                 offset += bar_width + group_spacing

#         ax.set_xticks(x_positions)
#         ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=9)
#         ax.set_ylabel("Fraction of Total Carbon Footprint")
#         ax.set_xlabel(f"Tiers / Placement (Cache={cache_size})")
#         ax.set_title(f"OPEX vs CAPEX Fraction per Tier – {topology} – α={alpha} – Cache={cache_size}")
#         ax.legend(title="CF Component", fontsize=9)
#         ax.grid(True, linestyle="--", alpha=0.6, axis="y")

#         ax.set_ylim(0, 1.05)
#         ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y*100)}%"))

#         plt.tight_layout()
#         outname = f"TIER_FRACTION_BY_PLACEMENT_T={topology}_A={alpha}_Cache={cache_size}_{mode}.jpg"
#         plt.savefig(os.path.join(plotdir, outname), bbox_inches="tight", dpi=300)
#         plt.close()
#         print(f"✅ Saved plot (fixed cache size): {outname}")

#     # ---------- (B) FIX PLACEMENT ----------
#     for placement in placements:
#         fig, ax = plt.subplots(figsize=(10, 5))
#         x_positions, xtick_labels = [], []
#         offset = 0

#         for cache_size in cache_size_range:
#             filtered = resultset.filter({
#                 "topology": {"name": topology},
#                 "cache_placement": {"network_cache": cache_size},
#                 "cache_placement": {"name": placement},
#                 "workload": {"name": "STATIONARY", "alpha": alpha},
#             })
#             if len(filtered) == 0:
#                 continue

#             result = filtered[0][1]
#             cf_data = result.get("CARBONFOOTPRINT", {})
#             per_tier = cf_data.get("PER_TIER", {})
#             tier_stats = cf_data.get("TIER_STATS", result.get("TIER_STATS", {}))

#             if not per_tier:
#                 continue

#             fractions = compute_fractions(per_tier, tier_stats)

#             for tier_name, (opex_frac, capex_frac) in fractions.items():
#                 ax.bar(offset, opex_frac, bar_width, color=cf_colors[0],
#                        hatch='/', edgecolor='black', label=cf_labels[0] if offset == 0 else "")
#                 ax.bar(offset, capex_frac, bar_width, bottom=opex_frac,
#                        color=cf_colors[1], hatch='\\', edgecolor='black', label=cf_labels[1] if offset == 0 else "")

#                 xtick_labels.append(f"{tier_name}\nCache {cache_size}")
#                 x_positions.append(offset)
#                 offset += bar_width + group_spacing

#         ax.set_xticks(x_positions)
#         ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=9)
#         ax.set_ylabel("Fraction of Total Carbon Footprint")
#         ax.set_xlabel(f"Tiers / Cache Size (Placement={placement})")
#         ax.set_title(f"OPEX vs CAPEX Fraction per Tier – {topology} – α={alpha} – Placement={placement}")
#         ax.legend(title="CF Component", fontsize=9)
#         ax.grid(True, linestyle="--", alpha=0.6, axis="y")

#         ax.set_ylim(0, 1.05)
#         ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y*100)}%"))

#         plt.tight_layout()
#         outname = f"TIER_FRACTION_BY_CACHE_T={topology}_A={alpha}_Placement={placement}_{mode}.jpg"
#         plt.savefig(os.path.join(plotdir, outname), bbox_inches="tight", dpi=300)
#         plt.close()
#         print(f"✅ Saved plot (fixed placement): {outname}")

# def plot_per_tier_opex_vs_capex_all_configs(
#     resultset, topology, alpha, cache_size_range, placements, plotdir
# ):
#     """
#     Plot stacked bars of DRAM/SSD OPEX and CAPEX per cache placement and size.
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import os

#     # Colors for (DRAM OPEX, SSD OPEX, DRAM CAPEX, SSD CAPEX)
#     colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
#     labels = ['DRAM OPEX', 'SSD OPEX', 'DRAM CAPEX', 'SSD CAPEX']
#     hatches = ['/', '\\', '...', 'xx']

#     fig, ax = plt.subplots(figsize=(12, 6))
#     bar_width = 0.35
#     spacing = 0.4

#     x_positions, xtick_labels = [], []

#     for i, cache_size in enumerate(cache_size_range):
#         for k, placement in enumerate(placements):
#             filtered = resultset.filter({
#                 "topology": {"name": topology},
#                 "cache_placement": {"network_cache": cache_size},
#                 "cache_placement": {"name": placement},
#                 "workload": {"name": "STATIONARY", "alpha": alpha},
#             })
#             cf = filtered[0][1]['CARBONFOOTPRINT'] if len(filtered) > 0 else {}

#             # Extract tier-level data
#             per_tier = cf.get('PER_TIER', {})
#             dram = per_tier.get('DRAM', {})
#             ssd = per_tier.get('SSD', {})

#             dram_opex = dram.get('OPEX', 0)
#             ssd_opex = ssd.get('OPEX', 0)
#             dram_capex = dram.get('CAPEX', 0)
#             ssd_capex = ssd.get('CAPEX', 0)

#             # Build the stack bottom-up
#             x = i * (len(placements) * (bar_width + spacing)) + k * (bar_width + spacing)
#             x_positions.append(x)
#             xtick_labels.append(f'{placement}\n(Cache {cache_size})')

#             bottoms = 0
#             for j, val in enumerate([dram_opex, ssd_opex, dram_capex, ssd_capex]):
#                 ax.bar(
#                     x,
#                     val,
#                     bar_width,
#                     bottom=bottoms,
#                     color=colors[j],
#                     hatch=hatches[j],
#                     edgecolor='black',
#                     label=labels[j] if (i == 0 and k == 0) else None
#                 )
#                 bottoms += val

#     # X-axis formatting
#     ax.set_xticks(x_positions)
#     ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=10)

#     # Labels and legend
#     ax.set_xlabel('Cache Proportion and Placement', fontsize=13)
#     ax.set_ylabel('Carbon Footprint (gCO₂e)', fontsize=13)
#     ax.legend(title="Component", fontsize=9, loc='upper right', ncol=2)
#     ax.grid(True, linestyle='--', alpha=0.6)

#     plt.tight_layout()
#     filename = os.path.join(plotdir, f"PER_TIER_STACKED_T={topology}@A={alpha}.jpg")
#     plt.savefig(filename, bbox_inches='tight')
#     plt.close()
#     print(f"✅ Saved stacked per-tier OPEX+CAPEX plot to {filename}")

# def plot_per_tier_opex_vs_capex_all_configs(
#     resultset, topology, alpha, cache_size_range, placements, plotdir,
#     n_contents=10000, data_size_range=(1000, 8000), baseline=("UNIFORM", 0.01)
# ):
#     """
#     Plot normalized (gCO2e/GB) per-tier OPEX/CAPEX with a secondary y-axis showing
#     efficiency improvement (%) relative to a chosen baseline configuration.
#     """

#     import numpy as np
#     import matplotlib.pyplot as plt
#     import os

#     # ----- Basic parameters -----
#     avg_content_size = sum(data_size_range) / 2  # bytes
#     total_cacheable_bytes = n_contents * avg_content_size

#     colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
#     labels = ['DRAM OPEX', 'SSD OPEX', 'DRAM CAPEX', 'SSD CAPEX']
#     hatches = ['/', '\\', '...', 'xx']

#     fig, ax1 = plt.subplots(figsize=(12, 6))
#     bar_width = 0.35
#     spacing = 0.4

#     x_positions, xtick_labels = [], []
#     cf_norm_values = {}  # for improvement computation

#     # ----- Compute normalized CF for each configuration -----
#     for i, cache_size in enumerate(cache_size_range):
#         for k, placement in enumerate(placements):
#             filtered = resultset.filter({
#                 "topology": {"name": topology},
#                 "cache_placement": {"network_cache": cache_size},
#                 "cache_placement": {"name": placement},
#                 "workload": {"name": "STATIONARY", "alpha": alpha},
#             })
#             cf = filtered[0][1].get('CARBONFOOTPRINT', {}) if len(filtered) > 0 else {}
#             per_tier = cf.get('PER_TIER', {})

#             dram = per_tier.get('DRAM', {})
#             ssd = per_tier.get('SSD', {})

#             dram_opex = dram.get('OPEX', 0)
#             ssd_opex = ssd.get('OPEX', 0)
#             dram_capex = dram.get('CAPEX', 0)
#             ssd_capex = ssd.get('CAPEX', 0)

#             # ---- Normalization factor ----
#             cache_bytes = cache_size * total_cacheable_bytes
#             cache_gb = cache_bytes / (1024 ** 3)
#             if cache_gb == 0:
#                 continue
#             norm = 1 / cache_gb

#             dram_opex *= norm
#             ssd_opex  *= norm
#             dram_capex *= norm
#             ssd_capex  *= norm

#             total_cf = dram_opex + ssd_opex + dram_capex + ssd_capex
#             cf_norm_values[(placement, cache_size)] = total_cf

#             # ---- Draw stacked bars ----
#             x = i * (len(placements) * (bar_width + spacing)) + k * (bar_width + spacing)
#             x_positions.append(x)
#             xtick_labels.append(f'{placement}\n(Cache {cache_size})')

#             bottom = 0
#             for j, val in enumerate([dram_opex, ssd_opex, dram_capex, ssd_capex]):
#                 ax1.bar(
#                     x, val, bar_width, bottom=bottom,
#                     color=colors[j], hatch=hatches[j], edgecolor='black',
#                     label=labels[j] if (i == 0 and k == 0) else None
#                 )
#                 bottom += val

#     # ----- Left Y-axis -----
#     ax1.set_ylabel('Carbon Footprint (g CO₂e / GB)', fontsize=12)
#     ax1.grid(True, linestyle='--', alpha=0.6)

#     # ----- Compute baseline and improvements -----
#     baseline_val = cf_norm_values.get(baseline, None)
#     if baseline_val is None:
#         print(f"⚠️ Baseline {baseline} not found — skipping efficiency axis.")
#         plt.close()
#         return

#     improvements = []
#     for key, cf in cf_norm_values.items():
#         imp = ((baseline_val - cf) / baseline_val) * 100
#         improvements.append((key, imp))

#     # Sort improvements to match x-axis order
#     improvement_values = [imp for ((p, c), imp) in sorted(improvements,
#                              key=lambda kv: (cache_size_range.index(kv[0][1]),
#                                              placements.index(kv[0][0])))]

#     # ----- Right Y-axis -----
#     ax2 = ax1.twinx()
#     ax2.plot(x_positions, improvement_values, 'ko--', markersize=4, label='Efficiency Δ (%)')
#     ax2.set_ylabel('Efficiency Improvement (%) vs Baseline', fontsize=12, color='black')
#     ax2.tick_params(axis='y', labelcolor='black')

#     # ----- X-axis -----
#     ax1.set_xticks(x_positions)
#     ax1.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=10)
#     ax1.set_xlabel('Cache Proportion and Placement', fontsize=13)
#     ax1.legend(title="Component", fontsize=9, loc='upper left', ncol=2)
#     ax2.legend(loc='upper right')

#     plt.tight_layout()
#     fname = os.path.join(plotdir, f"NORMALIZED_EFFICIENCY_T={topology}@A={alpha}.jpg")
#     plt.savefig(fname, bbox_inches='tight')
#     plt.close()
#     print(f"✅ Saved normalized efficiency plot: {fname}")

def plot_per_tier_opex_vs_capex_all_configs(
    resultset, topology, alpha, cache_size_range, placements, plotdir,
    n_contents=100000, data_size_range=(1000, 8000), baseline=("UNIFORM", 0.015)
):
    """
    Plot normalized (gCO2e/GB) per-tier OPEX/CAPEX (DRAM, SSD, HDD)
    with efficiency improvement curves relative to the baseline configuration.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # ---- Compute total cacheable bytes ----
    avg_content_size = np.mean(data_size_range)
    total_cacheable_bytes = n_contents * avg_content_size

    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467BD', '#E377C2']
    labels = ['DRAM OPEX', 'SSD OPEX', 'HDD OPEX', 'DRAM CAPEX', 'SSD CAPEX', 'HDD CAPEX']
    hatches = ['/', '\\', '...', 'xx', '||', '--']

    fig, ax1 = plt.subplots(figsize=(13, 6))
    bar_width = 0.35
    spacing = 0.4

    x_positions, xtick_labels = [], []
    cf_norm_total, cf_norm_dram, cf_norm_ssd, cf_norm_hdd = {}, {}, {}, {}

    # ---- Iterate over configurations ----
    for i, cache_size in enumerate(cache_size_range):
        for k, placement in enumerate(placements):
            filtered = resultset.filter({
                "topology": {"name": topology},
                "cache_placement": {"network_cache": cache_size, "name": placement},
                "workload": {"name": "STATIONARY", "alpha": alpha},
            })
            cf = filtered[0][1].get('CARBONFOOTPRINT', {}) if len(filtered) > 0 else {}
            per_tier = cf.get('PER_TIER', {})

            dram, ssd, hdd = per_tier.get('DRAM', {}), per_tier.get('SSD', {}), per_tier.get('HDD', {})

            dram_opex, dram_capex = dram.get('OPEX', 0), dram.get('CAPEX', 0)
            ssd_opex, ssd_capex = ssd.get('OPEX', 0), ssd.get('CAPEX', 0)
            hdd_opex, hdd_capex = hdd.get('OPEX', 0), hdd.get('CAPEX', 0)

            # ---- Normalize by cache capacity (in GB) ----
            cache_bytes = cache_size * total_cacheable_bytes
            cache_gb = cache_bytes / (1024 ** 3)
            if cache_gb == 0:
                continue

            norm_factor = 1.0 / cache_gb
            dram_opex, dram_capex = dram_opex * norm_factor, dram_capex * norm_factor
            ssd_opex, ssd_capex = ssd_opex * norm_factor, ssd_capex * norm_factor
            hdd_opex, hdd_capex = hdd_opex * norm_factor, hdd_capex * norm_factor

            dram_total = dram_opex + dram_capex
            ssd_total = ssd_opex + ssd_capex
            hdd_total = hdd_opex + hdd_capex
            total_cf = dram_total + ssd_total + hdd_total

            cf_norm_dram[(placement, cache_size)] = dram_total
            cf_norm_ssd[(placement, cache_size)] = ssd_total
            cf_norm_hdd[(placement, cache_size)] = hdd_total
            cf_norm_total[(placement, cache_size)] = total_cf

            # ---- Plot stacked bars ----
            x = i * (len(placements) * (bar_width + spacing)) + k * (bar_width + spacing)
            x_positions.append(x)
            xtick_labels.append(f'{placement}\n(Cache {cache_size})')

            bottom = 0
            for j, val in enumerate([dram_opex, ssd_opex, hdd_opex, dram_capex, ssd_capex, hdd_capex]):
                ax1.bar(
                    x, val, bar_width, bottom=bottom,
                    color=colors[j], hatch=hatches[j], edgecolor='black',
                    label=labels[j] if (i == 0 and k == 0) else None
                )
                bottom += val

    # ---- Left axis: Carbon footprint ----
    ax1.set_ylabel('Carbon Footprint (g CO₂e / GB)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # ---- Retrieve baseline ----
    baseline_val = cf_norm_total.get(baseline)
    baseline_dram = cf_norm_dram.get(baseline)
    baseline_ssd = cf_norm_ssd.get(baseline)
    baseline_hdd = cf_norm_hdd.get(baseline)

    if baseline_val is None:
        print(f"⚠️ Baseline {baseline} not found — skipping efficiency curves.")
        plt.close()
        return

    # ---- Compute efficiency improvements ----
    improvements_dram, improvements_ssd, improvements_hdd = [], [], []

    for key in cf_norm_total.keys():
        cf_dram, cf_ssd, cf_hdd = cf_norm_dram[key], cf_norm_ssd[key], cf_norm_hdd[key]
        imp_dram = ((baseline_dram - cf_dram) / baseline_dram) * 100 if baseline_dram > 0 else 0
        imp_ssd = ((baseline_ssd - cf_ssd) / baseline_ssd) * 100 if baseline_ssd > 0 else 0
        imp_hdd = ((baseline_hdd - cf_hdd) / baseline_hdd) * 100 if baseline_hdd > 0 else 0
        improvements_dram.append((key, imp_dram))
        improvements_ssd.append((key, imp_ssd))
        improvements_hdd.append((key, imp_hdd))

    # ---- Sort by cache and placement ----
    def sort_key(kv): return (cache_size_range.index(kv[0][1]), placements.index(kv[0][0]))
    improvements_dram = [imp for _, imp in sorted(improvements_dram, key=sort_key)]
    improvements_ssd = [imp for _, imp in sorted(improvements_ssd, key=sort_key)]
    improvements_hdd = [imp for _, imp in sorted(improvements_hdd, key=sort_key)]

    # ---- Right axis: Efficiency improvements ----
    ax2 = ax1.twinx()
    ax2.plot(x_positions, improvements_dram, 'b--o', label='DRAM Δ (%)', markersize=4)
    ax2.plot(x_positions, improvements_ssd, 'r--s', label='SSD Δ (%)', markersize=4)
    ax2.plot(x_positions, improvements_hdd, 'g--^', label='HDD Δ (%)', markersize=4)
    ax2.set_ylabel('Efficiency Improvement (%) vs Baseline', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='black')

    # ---- Axes and legends ----
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=10)
    ax1.set_xlabel('Cache Proportion and Placement', fontsize=13)
    ax1.legend(title="Component", fontsize=9, loc='upper left', ncol=2)
    ax2.legend(loc='upper right')

    # ---- Annotate best tier improvements ----
    def compute_best(cf_dict, baseline_value):
        if baseline_value and baseline_value > 0:
            best_key = max(cf_dict, key=lambda k: ((baseline_value - cf_dict[k]) / baseline_value))
            best_imp = ((baseline_value - cf_dict[best_key]) / baseline_value) * 100
            return best_key, best_imp
        return None, 0

    best_dram_key, best_dram_imp = compute_best(cf_norm_dram, baseline_dram)
    best_ssd_key, best_ssd_imp = compute_best(cf_norm_ssd, baseline_ssd)
    best_hdd_key, best_hdd_imp = compute_best(cf_norm_hdd, baseline_hdd)

    def key_to_xpos(key):
        return (cache_size_range.index(key[1]) * (len(placements) * (bar_width + spacing))
                + placements.index(key[0]) * (bar_width + spacing))

    for tier, key, imp, color, offset in [
        ('DRAM', best_dram_key, best_dram_imp, 'blue', 50),
        ('SSD', best_ssd_key, best_ssd_imp, 'red', -100),
        ('HDD', best_hdd_key, best_hdd_imp, 'green', -50)
    ]:
        if key:
            x_pos = key_to_xpos(key)
            ax2.annotate(
                f'Best {tier}\n{key[0]}@{key[1]} ({imp:.1f}%)',
                xy=(x_pos, imp), xytext=(x_pos, imp + offset),
                arrowprops=dict(arrowstyle='->', color=color),
                color=color, fontsize=9, ha='center'
            )

    # ---- Save plot ----
    plt.tight_layout()
    fname = os.path.join(plotdir, f"NORMALIZED_DUAL_EFFICIENCY_T={topology}@A={alpha}.jpg")
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved triple-efficiency plot (DRAM, SSD, HDD): {fname}")

# def plot_per_tier_opex_vs_capex_all_configs(
#     resultset,
#     topology,
#     alpha,
#     cache_size_range,
#     placements,
#     plotdir,
#     mode="global"  # 'global' or 'per-tier'
# ):
#     """
#     Generates stacked OPEX and CAPEX bars per configuration.
#     Each bar group = one configuration (placement or cache size).
#     Each bar is stacked by tier (DRAM, SSD, ...).
#     """
#     import os
#     import numpy as np
#     import matplotlib.pyplot as plt

#     # Color map for tiers
#     tier_colors = {
#         "DRAM": "#1f77b4",
#         "SSD": "#ff7f0e",
#         "NVDIMM": "#2ca02c",
#         "HDD": "#d62728",
#     }

#     bar_width = 0.35
#     group_spacing = 0.4

#     def compute_scaled(per_tier, tier_stats):
#         """Return tier-scaled OPEX/CAPEX dicts"""
#         if not tier_stats:
#             tier_stats = {t: 1 for t in per_tier.keys()}

#         scaled_opex = {t: v.get("OPEX", 0) * tier_stats.get(t, 1) for t, v in per_tier.items()}
#         scaled_capex = {t: v.get("CAPEX", 0) * tier_stats.get(t, 1) for t, v in per_tier.items()}

#         if mode == "global":
#             total = sum(list(scaled_opex.values()) + list(scaled_capex.values()))
#             if total > 0:
#                 scaled_opex = {t: v / total for t, v in scaled_opex.items()}
#                 scaled_capex = {t: v / total for t, v in scaled_capex.items()}
#         elif mode == "per-tier":
#             # normalize each tier internally
#             for t in per_tier.keys():
#                 total_tier = scaled_opex[t] + scaled_capex[t]
#                 if total_tier > 0:
#                     scaled_opex[t] /= total_tier
#                     scaled_capex[t] /= total_tier
#         return scaled_opex, scaled_capex

#     # ---------- (A) FIX CACHE SIZE ----------
#     for cache_size in cache_size_range:
#         fig, ax = plt.subplots(figsize=(10, 5))
#         x_positions, xtick_labels = [], []
#         offset = 0

#         for placement in placements:
#             filtered = resultset.filter({
#                 "topology": {"name": topology},
#                 "cache_placement": {"network_cache": cache_size},
#                 "cache_placement": {"name": placement},
#                 "workload": {"name": "STATIONARY", "alpha": alpha},
#             })
#             if len(filtered) == 0:
#                 continue

#             result = filtered[0][1]
#             cf_data = result.get("CARBONFOOTPRINT", {})
#             per_tier = cf_data.get("PER_TIER", {})
#             tier_stats = cf_data.get("TIER_STATS", result.get("TIER_STATS", {}))

#             if not per_tier:
#                 continue

#             scaled_opex, scaled_capex = compute_scaled(per_tier, tier_stats)

#             # OPEX bar (stacked)
#             bottom_opex = 0
#             for t, val in scaled_opex.items():
#                 ax.bar(offset, val, bar_width,
#                        bottom=bottom_opex,
#                        color=tier_colors.get(t, None),
#                        edgecolor='black',
#                        label=f"{t} OPEX" if offset == 0 else "")
#                 bottom_opex += val

#             # CAPEX bar (stacked)
#             bottom_capex = 0
#             for t, val in scaled_capex.items():
#                 ax.bar(offset + bar_width + 0.05, val, bar_width,
#                        bottom=bottom_capex,
#                        color=tier_colors.get(t, None),
#                        edgecolor='black',
#                        hatch='//',
#                        label=f"{t} CAPEX" if offset == 0 else "")
#                 bottom_capex += val

#             xtick_labels.append(f"{placement}")
#             x_positions.append(offset + bar_width / 2)
#             offset += (2 * bar_width) + group_spacing

#         ax.set_xticks(x_positions)
#         ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=9)
#         ax.set_ylabel("Fraction of Total Carbon Footprint")
#         ax.set_xlabel(f"Placement (Cache={cache_size})")
#         ax.set_title(f"Stacked OPEX vs CAPEX per Tier – {topology} – α={alpha} – Cache={cache_size}")
#         ax.grid(True, linestyle="--", alpha=0.6, axis="y")

#         ax.set_ylim(0, 1.05)
#         ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y*100)}%"))

#         handles, labels = ax.get_legend_handles_labels()
#         unique = dict(zip(labels, handles))
#         ax.legend(unique.values(), unique.keys(), title="Tier/Component", fontsize=8)

#         plt.tight_layout()
#         outname = f"STACKED_TIER_COMPONENTS_T={topology}_A={alpha}_Cache={cache_size}_{mode}.jpg"
#         plt.savefig(os.path.join(plotdir, outname), bbox_inches="tight", dpi=300)
#         plt.close()
#         print(f"✅ Saved stacked plot (Cache={cache_size}): {outname}")

# def plot_per_tier_opex_vs_capex_all_configs(resultset, topology, alpha, cache_size_range, placements, plotdir):
#     import os
#     import matplotlib.pyplot as plt

#     tier_colors = {
#         "DRAM": "#1f77b4",
#         "SSD": "#ff7f0e",
#         "NVDIMM": "#2ca02c",
#         "HDD": "#d62728",
#     }

#     bar_width = 0.35
#     group_spacing = 0.4

#     for cache_size in cache_size_range:
#         fig, ax = plt.subplots(figsize=(10, 5))
#         x_positions, xtick_labels = [], []
#         offset = 0

#         for placement in placements:
#             filtered = resultset.filter({
#                 "topology": {"name": topology},
#                 "cache_placement": {"network_cache": cache_size},
#                 "cache_placement": {"name": placement},
#                 "workload": {"name": "STATIONARY", "alpha": alpha},
#             })
#             if not filtered:
#                 continue

#             result = filtered[0][1]
#             cf_data = result.get("CARBONFOOTPRINT", {})
#             per_tier = cf_data.get("PER_TIER", {})
#             tier_stats = cf_data.get("TIER_STATS", {})

#             if not per_tier:
#                 continue

#             # Scale by number of nodes per tier
#             scaled_opex = {t: v.get("OPEX", 0) * tier_stats.get(t, 1) for t, v in per_tier.items()}
#             scaled_capex = {t: v.get("CAPEX", 0) * tier_stats.get(t, 1) for t, v in per_tier.items()}

#             total_opex = sum(scaled_opex.values())
#             total_capex = sum(scaled_capex.values())

#             # Normalize each group separately to sum to 1 within each component
#             if total_opex > 0:
#                 scaled_opex = {t: v / total_opex for t, v in scaled_opex.items()}
#             if total_capex > 0:
#                 scaled_capex = {t: v / total_capex for t, v in scaled_capex.items()}

#             # --- Draw OPEX (stacked)
#             bottom = 0
#             for t, val in scaled_opex.items():
#                 ax.bar(offset, val, bar_width, bottom=bottom,
#                        color=tier_colors.get(t, "#999"),
#                        edgecolor='black', hatch='//',
#                        label=f"{t} OPEX" if offset == 0 else "")
#                 if val > 0.01:
#                     ax.text(offset, bottom + val/2, f"{val*100:.1f}%", ha='center', va='center', fontsize=7)
#                 bottom += val

#             # --- Draw CAPEX (stacked)
#             bottom = 0
#             for t, val in scaled_capex.items():
#                 ax.bar(offset + bar_width + 0.05, val, bar_width, bottom=bottom,
#                        color=tier_colors.get(t, "#999"),
#                        edgecolor='black', hatch='\\\\',
#                        label=f"{t} CAPEX" if offset == 0 else "")
#                 if val > 0.01:
#                     ax.text(offset + bar_width + 0.05, bottom + val/2, f"{val*100:.1f}%", ha='center', va='center', fontsize=7)
#                 bottom += val

#             xtick_labels.append(f"{placement}")
#             x_positions.append(offset + bar_width / 2)
#             offset += (2 * bar_width) + group_spacing

#         ax.set_xticks(x_positions)
#         ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=9)
#         ax.set_ylabel("Fraction (normalized within OPEX/CAPEX)")
#         ax.set_xlabel(f"Placement (Cache={cache_size})")
#         ax.set_title(f"Stacked OPEX vs CAPEX per Tier – {topology} – α={alpha} – Cache={cache_size}")
#         ax.grid(True, linestyle="--", alpha=0.6, axis="y")

#         ax.set_ylim(0, 1.05)
#         ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y*100)}%"))

#         handles, labels = ax.get_legend_handles_labels()
#         unique = dict(zip(labels, handles))
#         ax.legend(unique.values(), unique.keys(), title="Tier/Component", fontsize=8)

#         plt.tight_layout()
#         outname = f"STACKED_TIER_COMPONENTS_FIXED_T={topology}_A={alpha}_Cache={cache_size}.jpg"
#         plt.savefig(os.path.join(plotdir, outname), bbox_inches="tight", dpi=300)
#         plt.close()
#         print(f"✅ Saved fixed stacked plot: {outname}")

# def plot_per_tier_opex_vs_capex_all_configs(resultset, topology, alpha, cache_size_range, placements, plotdir):
#     import os
#     import matplotlib.pyplot as plt

#     # --- Define tier colors ---
#     tier_colors = {
#         "DRAM": "#1f77b4",
#         "SSD": "#ff7f0e",
#         "NVDIMM": "#2ca02c",
#         "HDD": "#d62728",
#     }

#     bar_width = 0.35
#     group_spacing = 0.4

#     for cache_size in cache_size_range:
#         fig, ax = plt.subplots(figsize=(10, 5))
#         x_positions, xtick_labels = [], []
#         offset = 0

#         for placement in placements:
#             filtered = resultset.filter({
#                 "topology": {"name": topology},
#                 "cache_placement": {"network_cache": cache_size},
#                 "cache_placement": {"name": placement},
#                 "workload": {"name": "STATIONARY", "alpha": alpha},
#             })
#             if not filtered:
#                 continue

#             result = filtered[0][1]
#             cf_data = result.get("CARBONFOOTPRINT", {})
#             per_tier = cf_data.get("PER_TIER", {})
#             tier_stats = cf_data.get("TIER_STATS", {})
#             duration = cf_data.get("duration", 0) or 1  # default fallback

#             if not per_tier:
#                 continue

#             # -----------------------------
#             # Scale by node count and lifetime
#             # -----------------------------
#             scaled_opex, scaled_capex = {}, {}

#             for tier, vals in per_tier.items():
#                 opex = vals.get("OPEX", 0)
#                 capex = vals.get("CAPEX", 0)
#                 n_nodes = tier_stats.get(tier, 1)
#                 lifespan_years = 5 if "DRAM" in tier else (3 if "SSD" in tier else 5)
#                 lifespan_sec = lifespan_years * 365 * 24 * 60 * 60
#                 # Lifetime-aware scaling of CAPEX
#                 scaled_opex[tier] = opex * n_nodes
#                 scaled_capex[tier] = capex * n_nodes * (duration / lifespan_sec)

#             total_opex = sum(scaled_opex.values())
#             total_capex = sum(scaled_capex.values())

#             # Normalize to within OPEX or CAPEX
#             if total_opex > 0:
#                 scaled_opex = {t: v / total_opex for t, v in scaled_opex.items()}
#             if total_capex > 0:
#                 scaled_capex = {t: v / total_capex for t, v in scaled_capex.items()}

#             # -----------------------------
#             # Plot OPEX (stacked)
#             # -----------------------------
#             bottom = 0
#             for t, val in scaled_opex.items():
#                 ax.bar(offset, val, bar_width, bottom=bottom,
#                        color=tier_colors.get(t, "#999"),
#                        edgecolor='black', hatch='//',
#                        label=f"{t} OPEX" if offset == 0 else "")
#                 if val > 0.01:
#                     ax.text(offset, bottom + val / 2, f"{val*100:.1f}%", ha='center', va='center', fontsize=7)
#                 bottom += val

#             # -----------------------------
#             # Plot CAPEX (stacked)
#             # -----------------------------
#             bottom = 0
#             for t, val in scaled_capex.items():
#                 ax.bar(offset + bar_width + 0.05, val, bar_width, bottom=bottom,
#                        color=tier_colors.get(t, "#999"),
#                        edgecolor='black', hatch='\\\\',
#                        label=f"{t} CAPEX" if offset == 0 else "")
#                 if val > 0.01:
#                     ax.text(offset + bar_width + 0.05, bottom + val / 2, f"{val*100:.1f}%", ha='center', va='center', fontsize=7)
#                 bottom += val

#             xtick_labels.append(f"{placement}")
#             x_positions.append(offset + bar_width / 2)
#             offset += (2 * bar_width) + group_spacing

#         # -----------------------------
#         # Styling
#         # -----------------------------
#         ax.set_xticks(x_positions)
#         ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=9)
#         ax.set_ylabel("Fraction (normalized within OPEX/CAPEX)")
#         ax.set_xlabel(f"Placement (Cache={cache_size})")
#         ax.set_title(f"Stacked OPEX vs CAPEX per Tier – {topology} – α={alpha} – Cache={cache_size}")
#         ax.grid(True, linestyle="--", alpha=0.6, axis="y")

#         ax.set_ylim(0, 1.05)
#         ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y*100)}%"))

#         # Legend (deduplicated)
#         handles, labels = ax.get_legend_handles_labels()
#         unique = dict(zip(labels, handles))
#         ax.legend(unique.values(), unique.keys(), title="Tier/Component", fontsize=8)

#         plt.tight_layout()
#         outname = f"STACKED_TIER_COMPONENTS_LIFETIME_T={topology}_A={alpha}_Cache={cache_size}.jpg"
#         plt.savefig(os.path.join(plotdir, outname), bbox_inches="tight", dpi=300)
#         plt.close()
#         print(f"✅ Saved lifetime-scaled stacked plot: {outname}")

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
    # cache_items = settings.NETWORK_CACHE
    # original_list = []
    # geant 19 caches
    # garr 27 caches
    # rocket fuel 36 caches
    # wide 13 caches
    # for entry, metrics in resultset:
    #     cache_size = entry.get("cache_placement", {})["network_cache"]
    #     workload = entry.get("workload", {}).get("n_contents")
    #     normalized_value = int(cache_size * workload / 19)
    #     entry.get("cache_placement", {})["network_cache"] = normalized_value
    #     original_list.append(normalized_value)
    #     print(normalized_value)
    
    # cache_sizes = list(dict.fromkeys(original_list))
    cache_sizes = settings.NETWORK_CACHE
    strategies = settings.STRATEGIES
    cache_placements = settings.CACHE_PLACEMENT
    # Reorder to put CLS2M first if it's in the list
    if "CLS2M" in strategies:
        strategies = ["CLS2M"] + [s for s in strategies if s != "CLS2M"]
    alphas = settings.ALPHA
    # Plot graphs
    distributions = {
        "CL2SM": {1: 4520, 2: 66, 3: 57, 4: 4, 5: 6, 6: 1, 7: 1, 8: 4, 9: 1, 10: 2},
        "LCE": {2: 192, 3: 812, 4: 1222, 5: 1112, 6: 964, 7: 350, 8: 6, 9: 2, 10: 1, 11: 1},
        "LCD": {2: 4590, 3: 45, 4: 11, 5: 6, 6: 4, 7: 3, 8: 1, 9: 1, 10: 1},
        "PROB_CACHE": {1: 3373, 2: 1191, 3: 90, 4: 8, 6: 2, 7: 1, 8: 2, 9: 2},
        "CPCache": {1: 183, 2: 792, 3: 1206, 4: 1110, 5: 974, 6: 368, 7: 20, 8: 6, 9: 1, 10: 2},
        "CL4M": {2: 4617, 3: 31, 4: 5, 5: 2, 6: 1, 8: 2, 9: 2, 10: 2},
    }
    
    for topology in topologies:
        for alpha in alphas:
            for strategy in strategies:
                plot_cf_vs_cache_placement(
                    resultset, topology, alpha, cache_sizes, strategy, cache_placements, plotdir
                )
                plot_capex_vs_cache_placement(
                    resultset, topology, alpha, cache_sizes, strategy, cache_placements, plotdir
                )
                plot_opex_vs_cache_placement(
                    resultset, topology, alpha, cache_sizes, strategy, cache_placements, plotdir
                )
                plot_cache_hits_vs_cache_placement(
                    resultset, topology, alpha, cache_sizes, strategy, cache_placements, plotdir
                )
                plot_cost_vs_cache_placement(
                    resultset, topology, alpha, cache_sizes, strategy, cache_placements, plotdir
                )
                plot_chrcp_vs_cache_placement(
                    resultset, topology, alpha, cache_sizes, strategy, cache_placements, plotdir
                )
                plot_latency_vs_cache_placement(
                    resultset, topology, alpha, cache_sizes, strategy, cache_placements, plotdir
                )
                # plot_link_load_vs_cache_placement(
                #     resultset, topology, alpha, cache_sizes, strategy, cache_placements, plotdir
                # ) 
            plot_opex_vs_capex_cache_size(
                resultset, topology, alpha, cache_sizes, cache_placements, plotdir
            )
            plot_per_tier_opex_vs_capex_all_configs(
                resultset, topology, alpha, cache_sizes, cache_placements, plotdir
            )
            plot_network_cf_components_vs_cache_size(
                resultset, topology, alpha, cache_sizes, cache_placements, plotdir
            )
            plot_server_opex_vs_capex_cache_size(
                resultset, topology, alpha, cache_sizes, cache_placements, plotdir
            )
            plot_cache_hits_vs_cache_size(
                resultset, topology, alpha, cache_sizes, strategies, plotdir
            )
            plot_chrcp_vs_cache_size(
                resultset, topology, alpha, cache_sizes, strategies, plotdir
            )
            plot_latency_vs_cache_size(
                resultset, topology, alpha, cache_sizes, strategies, plotdir
            )
            plot_cost_vs_cache_size(
                resultset, topology, alpha, cache_sizes, strategies, plotdir
            )
            plot_cost_components_vs_cache_size(
            resultset, topology, alpha, cache_sizes, strategies, plotdir
            )
        for cache_size in cache_sizes:
            plot_cache_hits_vs_alpha(
                resultset, topology, cache_size, alphas, strategies, plotdir
            )
            plot_latency_vs_alpha(
                resultset, topology, cache_size, alphas, strategies, plotdir
            )
            plot_cost_vs_alpha(
                resultset, topology, cache_size, alphas, strategies, plotdir
            )
            plot_chrcp_vs_alpha(
                resultset, topology, cache_size, alphas, strategies, plotdir
            )
            plot_cost_components_alpha(
                resultset, topology, cache_size, alphas, strategies, plotdir
            )
        
    plot_num_contents_vs_replicas(distributions, plotdir="./plots")

    for cache_size in cache_sizes:
        for alpha in alphas:
            plot_cache_hits_vs_topology(
                resultset, alpha, cache_size, topologies, strategies, plotdir
            )
            plot_latency_vs_topology(
                resultset, alpha, cache_size, topologies, strategies, plotdir
            )
            plot_chrcp_vs_topology(
                resultset, alpha, cache_size, topologies, strategies, plotdir
            )
            plot_cost_vs_topology(
                resultset, alpha, cache_size, topologies, strategies, plotdir
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