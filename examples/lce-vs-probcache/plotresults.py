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

def plot_storage_cf_components_vs_cache_size(
    resultset, topology, alpha, cache_size_range, placements, plotdir
):
    """
    Plot carbon footprint components for each placement as a stacked bar plot with placement names under each bar.
    """
    # Cost component names in the result set
    cf_components = ["DEPRECIATION_CF", "READ_CF", "WRITE_CF", "EMBODIED_CF", "IDLE_CF"]
    num_components = len(cf_components)
    
    # Prepare for plotting
    fig, ax = plt.subplots(figsize=(10, 6))  # Increase figure size
    bar_width = 0.15  # Width of each placement's bar
    bar_spacing = 0.05  # Extra space between groups of bars
    total_bars_per_group = len(placements) * (bar_width + bar_spacing)
    
    # Generate positions for each bar, spacing them based on both cache sizes and strategies
    x_positions = []
    for i, cache_size in enumerate(cache_size_range):
        for j, placement in enumerate(placements):
            x_positions.append(i * (total_bars_per_group + 0.2) + j * (bar_width + bar_spacing))
    
    x_positions = np.array(x_positions)
    
    # Define color and hatch styles for each cost component
    cf_colors = ['#FF7F0E', '#1F77B4', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2']
    cf_hatches = ['/', '\\', '|', '-', '+', 'x', 'o']
    
    # Plot bars for each placement and component
    for i, placement in enumerate(placements):
        bottom = np.zeros(len(cache_size_range))  # Initialize for stacking bars

        for j, component in enumerate(cf_components):
            data = []
            for cache_size in cache_size_range:
                filtered = resultset.filter({
                    "topology": {"name": topology},
                    "cache_placement": {"network_cache": cache_size},
                    "cache_placement": {"name": placement},
                    "workload" :{"name": "STATIONARY", "alpha": alpha},
                })
                
                cf = filtered[0][1]['CARBONFOOTPRINT'].get(component, 0) if len(filtered) > 0 else 0
                data.append(cf)
            
            ax.bar(
                x_positions[i::len(placements)], data, bar_width,
                bottom=bottom,
                color=cf_colors[j],
                hatch=cf_hatches[j]
            )
            bottom += np.array(data)

    # Set labels, title, ticks, and legends
    ax.set_xlabel('Cache Proportion and Placement', fontsize=14)
    ax.set_ylabel('Carbon Footprint', fontsize=14)
    
    # Add cache size and placement labels as x-axis labels
    xtick_labels = []
    for cache_size in cache_size_range:
        for placement in placements:
            xtick_labels.append(f'{placement}\n(Cache {cache_size})')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xtick_labels, fontsize=10, rotation=45, ha="right")
    
    # Add gridlines
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Add a legend for the cost components only
    handles = [plt.Rectangle((0,0),1,1, color=cf_colors[i], hatch=cf_hatches[i]) for i in range(num_components)]
    ax.legend(handles, cf_components, loc='upper right', fontsize=10, title="CF Components")

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir, f"STORAGE_CF_T={topology}@A={alpha}.jpg"), bbox_inches='tight')

def plot_network_cf_components_vs_cache_size(
    resultset, topology, alpha, cache_size_range, placements, plotdir
):
    """
    Plot carbon footprint components for each placement as a stacked bar plot with placement names under each bar.
    """
    # Cost component names in the result set
    cf_components = ["ROUTERS_CF", "LINKS_CF"]
    num_components = len(cf_components)
    
    # Prepare for plotting
    fig, ax = plt.subplots(figsize=(10, 6))  # Increase figure size
    bar_width = 0.15  # Width of each placement's bar
    bar_spacing = 0.05  # Extra space between groups of bars
    total_bars_per_group = len(placements) * (bar_width + bar_spacing)
    
    # Generate positions for each bar, spacing them based on both cache sizes and strategies
    x_positions = []
    for i, cache_size in enumerate(cache_size_range):
        for j, placement in enumerate(placements):
            x_positions.append(i * (total_bars_per_group + 0.2) + j * (bar_width + bar_spacing))
    
    x_positions = np.array(x_positions)
    
    # Define color and hatch styles for each cost component
    cf_colors = ['#FF7F0E', '#1F77B4', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2']
    cf_hatches = ['/', '\\', '|', '-', '+', 'x', 'o']
    
    # Plot bars for each placement and component
    for i, placement in enumerate(placements):
        bottom = np.zeros(len(cache_size_range))  # Initialize for stacking bars

        for j, component in enumerate(cf_components):
            data = []
            for cache_size in cache_size_range:
                filtered = resultset.filter({
                    "topology": {"name": topology},
                    "cache_placement": {"network_cache": cache_size},
                    "cache_placement": {"name": placement},
                    "workload" :{"name": "STATIONARY", "alpha": alpha},
                })
                
                cf = filtered[0][1]['CARBONFOOTPRINT'].get(component, 0) if len(filtered) > 0 else 0
                data.append(cf)
            
            ax.bar(
                x_positions[i::len(placements)], data, bar_width,
                bottom=bottom,
                color=cf_colors[j],
                hatch=cf_hatches[j]
            )
            bottom += np.array(data)

    # Set labels, title, ticks, and legends
    ax.set_xlabel('Cache Proportion and Placement', fontsize=14)
    ax.set_ylabel('Carbon Footprint', fontsize=14)
    
    # Add cache size and placement labels as x-axis labels
    xtick_labels = []
    for cache_size in cache_size_range:
        for placement in placements:
            xtick_labels.append(f'{placement}\n(Cache {cache_size})')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xtick_labels, fontsize=10, rotation=45, ha="right")
    
    # Add gridlines
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Add a legend for the cost components only
    handles = [plt.Rectangle((0,0),1,1, color=cf_colors[i], hatch=cf_hatches[i]) for i in range(num_components)]
    ax.legend(handles, cf_components, loc='upper right', fontsize=10, title="CF Components")

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir, f"NETWORK_CF_T={topology}@A={alpha}.jpg"), bbox_inches='tight')

def plot_opex_cf_vs_capex_cf_cache_size(
    resultset, topology, alpha, cache_size_range, placements, plotdir
):
    """
    Plot carbon footprint components (OPEX vs CAPEX) for each placement and cache size
    as side-by-side bars (not stacked).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # Define cost components
    opex_capex = ["TOTAL_OPEX", "TOTAL_CAPEX"]
    num_components = len(opex_capex)
    
    # Basic layout
    fig, ax = plt.subplots(figsize=(11, 6))
    bar_width = 0.18
    group_spacing = 0.4  # space between cache size groups

    cf_colors = ['#FF7F0E', '#1F77B4']
    cf_hatches = ['/', '\\']

    # Compute base x positions per cache size and placement
    total_groups = len(cache_size_range)
    total_placements = len(placements)
    indices = np.arange(total_groups * total_placements)

    # shift offset between OPEX and CAPEX
    component_offset = bar_width + 0.02

    # --- Plot ---
    for j, component in enumerate(opex_capex):
        x_positions = []
        data = []
        for i, cache_size in enumerate(cache_size_range):
            for k, placement in enumerate(placements):
                filtered = resultset.filter({
                    "topology": {"name": topology},
                    "cache_placement": {"network_cache": cache_size},
                    "cache_placement": {"name": placement},
                    "workload": {"name": "STATIONARY", "alpha": alpha},
                })
                cf = filtered[0][1]['CARBONFOOTPRINT'].get(component, 0) if len(filtered) > 0 else 0
                data.append(cf)
                x_positions.append(i * (len(placements) * (num_components * bar_width + 0.2))
                                   + k * (num_components * bar_width + 0.1)
                                   + j * component_offset)

        ax.bar(
            x_positions,
            data,
            bar_width,
            label=component,
            color=cf_colors[j],
            hatch=cf_hatches[j],
            edgecolor='black'
        )

    # --- X-axis labels ---
    xtick_labels = []
    xtick_positions = []
    for i, cache_size in enumerate(cache_size_range):
        for k, placement in enumerate(placements):
            base_x = i * (len(placements) * (num_components * bar_width + 0.2)) + k * (num_components * bar_width + 0.1) + (bar_width / 2)
            xtick_positions.append(base_x + component_offset / 2)
            xtick_labels.append(f'{placement}\n(Cache {cache_size})')

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=10)

    # --- Labels and style ---
    ax.set_xlabel('Cache Proportion and Placement', fontsize=13)
    ax.set_ylabel('Carbon Footprint (kgCO₂e)', fontsize=13)
    ax.legend(title="CF Component", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(plotdir, f"CAPEX_VS_OPEX_T={topology}@A={alpha}.jpg"), bbox_inches='tight')
    plt.close()


def plot_cf_vs_cache_size(
    resultset, topology, alpha, cache_size_range, strategies, plotdir
):
    desc = {}
    if "NO_CACHE" in strategies:
        strategies.remove("NO_CACHE")
    # print("here")
    desc["xlabel"] = "Cache Proportion (%)"
    desc["ylabel"] = "Carbon Footprint Kg.CO2"
    desc["xparam"] = ("cache_placement", "network_cache")
    desc["xvals"] = cache_size_range
    desc["filter"] = {
        "topology": {"name": topology},
        "workload": {"name": "STATIONARY", "alpha": alpha},
    }
    desc["ymetrics"] = [("CARBONFOOTPRINT", "MEAN")] * len(strategies)
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
        "CARBONFOOTPRINT_T={}@A={}.jpg".format(topology, alpha),
        plotdir,
    )

def plot_em_cf_vs_cache_size(
    resultset, topology, alpha, cache_size_range, strategies, plotdir
):
    desc = {}
    if "NO_CACHE" in strategies:
        strategies.remove("NO_CACHE")
    # print("here")
    desc["xlabel"] = "Cache Proportion (%)"
    desc["ylabel"] = "Embodied Carbon Footprint Kg.CO2"
    desc["xparam"] = ("cache_placement", "network_cache")
    desc["xvals"] = cache_size_range
    desc["filter"] = {
        "topology": {"name": topology},
        "workload": {"name": "STATIONARY", "alpha": alpha},
    }
    desc["ymetrics"] = [("CARBONFOOTPRINT", "EMBODIED_CF")] * len(strategies)
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
        "EMBODIEDCARBONFOOTPRINT_T={}@A={}.jpg".format(topology, alpha),
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
    desc = {}
    desc["xlabel"] = "Cache Placements"
    desc["ylabel"] = "Cost per Request ($)"
    desc["xparam"] = ("cache_placement", "network_cache")
    desc["xvals"] = cache_size_range
    desc["filter"] = {
        "topology": {"name": topology},
        "workload": {"name": "STATIONARY", "alpha" : alpha},
        "strategy": {"name": strategy},
    }
    desc["ymetrics"] = [("COST", "MEAN")] * len(cache_placements)
    desc["ycondnames"] = [("cache_placement", "name")] * len(cache_placements)
    desc["ycondvals"] = cache_placements
    desc["metric"] = ("COST", "MEAN")
    desc["errorbar"] = True
    desc["legend_loc"] = "upper right"
    desc["line_style"] = PLACEMENT_STYLE
    desc["legend"] = PLACEMENT_LEGEND
    desc["plotempty"] = PLOT_EMPTY_GRAPHS
    plot_lines(
        resultset, desc, "COST_T={}@A={}@S={}.jpg".format(topology, alpha, strategy), plotdir
    )

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
    desc = {}
    desc["xlabel"] = "Cache Placements"
    desc["ylabel"] = "Carbon Footprint Kg.CO2"
    desc["xparam"] = ("cache_placement", "network_cache")
    desc["xvals"] = cache_size_range
    desc["filter"] = {
        "topology": {"name": topology},
        "workload": {"name": "STATIONARY", "alpha": alpha},
        "strategy": {"name": strategy},
    }
    desc["ymetrics"] = [("CARBONFOOTPRINT", "MEAN")] * len(cache_placements)
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
        "CARBONFOOTPRINT_T={}@A={}@S={}.jpg".format(topology, alpha, strategy),
        plotdir,
    )

def plot_em_cf_vs_cache_placement(
    resultset, topology, alpha, cache_size_range, strategy, cache_placements, plotdir
):
    desc = {}
    desc["xlabel"] = "Cache Placements"
    desc["ylabel"] = "Embodied Carbon Footprint Kg.CO2"
    desc["xparam"] = ("cache_placement", "network_cache")
    desc["xvals"] = cache_size_range
    desc["filter"] = {
        "topology": {"name": topology},
        "workload": {"name": "STATIONARY", "alpha": alpha},
        "strategy": {"name": strategy},
    }
    desc["ymetrics"] = [("CARBONFOOTPRINT", "EMBODIED_CF")] * len(cache_placements)
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
        "EMBODIEDCARBONFOOTPRINT_T={}@A={}@S={}.jpg".format(topology, alpha, strategy),
        plotdir,
    )

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
                plot_cf_vs_cache_size(
                    resultset, topology, alpha, cache_sizes, strategies, plotdir
                )
                plot_cf_vs_cache_placement(
                    resultset, topology, alpha, cache_sizes, strategy, cache_placements, plotdir
                )
                plot_em_cf_vs_cache_placement(
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
            plot_storage_cf_components_vs_cache_size(
                resultset, topology, alpha, cache_sizes, cache_placements, plotdir
            )
            plot_network_cf_components_vs_cache_size(
                resultset, topology, alpha, cache_sizes, cache_placements, plotdir
            )
            plot_opex_cf_vs_capex_cf_cache_size(
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
        # for cache_size in cache_sizes:
        #     plot_cache_hits_vs_alpha(
        #         resultset, topology, cache_size, alphas, strategies, plotdir
        #     )
        #     plot_latency_vs_alpha(
        #         resultset, topology, cache_size, alphas, strategies, plotdir
        #     )
        #     plot_cost_vs_alpha(
        #         resultset, topology, cache_size, alphas, strategies, plotdir
        #     )
        #     plot_chrcp_vs_alpha(
        #         resultset, topology, cache_size, alphas, strategies, plotdir
        #     )
        #     plot_cost_components_alpha(
        #         resultset, topology, cache_size, alphas, strategies, plotdir
        #     )
        
    # plot_num_contents_vs_replicas(distributions, plotdir="./plots")

    # for cache_size in cache_sizes:
    #     for alpha in alphas:
    #         plot_cache_hits_vs_topology(
    #             resultset, alpha, cache_size, topologies, strategies, plotdir
    #         )
    #         plot_latency_vs_topology(
    #             resultset, alpha, cache_size, topologies, strategies, plotdir
    #         )
    #         plot_chrcp_vs_topology(
    #             resultset, alpha, cache_size, topologies, strategies, plotdir
    #         )
    #         plot_cost_vs_topology(
    #             resultset, alpha, cache_size, topologies, strategies, plotdir
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