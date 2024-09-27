"""This module contains all configuration information used to run simulations"""
from multiprocessing import cpu_count
from collections import deque
import copy
from icarus.util import Tree

# GENERAL SETTINGS

# Level of logging output
# Available options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = "INFO"

# If True, executes simulations in parallel using multiple processes
# to take advantage of multicore CPUs
PARALLEL_EXECUTION = False

# Number of processes used to run simulations in parallel.
# This option is ignored if PARALLEL_EXECUTION = False
N_PROCESSES = cpu_count()

# Granularity of caching.
# Currently, only OBJECT is supported
CACHING_GRANULARITY = "OBJECT"

# Format in which results are saved.
# Result readers and writers are located in module ./icarus/results/readwrite.py
# Currently only PICKLE is supported
RESULTS_FORMAT = "PICKLE"

# Number of times each experiment is replicated
N_REPLICATIONS = 1

# List of metrics to be measured in the experiments
# The implementation of data collectors are located in ./icarus/execution/collectors.py
# DATA_COLLECTORS = ["CACHE_HIT_RATIO", "LATENCY", "COST"]


TIERS = [
    {"name":"DRAM",
    "size_factor": 1 / 31,
    "purchase_cost" : 150, # in $
    "lifespan" : 5, # in years
    "read_throughput" : 4e+10,  # 40GBPS
    "write_throughput" : 2e+10, # 20GBPS
    "latency"  : 1e-7,  #100ns
    "active_caching_power_density" : 10**-9,  # w/bit
    "idle_power_density" : 10**-9,  # w/bit
    }, 
    {"name":"SSD",
    "size_factor": 5 / 31,
    "purchase_cost" : 118, # in $
    "lifespan" : 5, # in years
    "read_throughput" : 3e+9,  # 3GBPS
    "write_throughput" : 1e+9, # 1GBPS
    "latency"  : 1e-5,  #10000ns
    "active_caching_power_density" : 10**-9,  # w/bit
    "idle_power_density" : 10**-9,  # w/bit
    }, 
    {"name":"HDD",
    "size_factor": 25 / 31,
    "purchase_cost" : 65, # in $
    "lifespan" : 5, # in years
    "read_throughput" : 1.5e+8,  #150MBPS
    "write_throughput" : 3e+7, # 30MBPS
    "latency"  : 4e-3,  #4ms
    "active_caching_power_density" : 10**-9,  # w/bit
    "idle_power_density" : 10**-9,  # w/bit
    }]

# Create tree of experiment configuration
default = Tree()

# Specify workload
default["workload"] = {
    "name": "STATIONARY",
    "alpha": 1.2,
    "n_contents": 3 * 10 ** 3,
    "n_warmup": 3 * 10 ** 3,
    "n_measured": 6 * 10 ** 3,
    "rate": 12,
    "high_priority_rate" :0.2,
    "priority_values": ["low", "high"],
    "data_size_range" : [1024, 4096]
}

# Specify cache placement
# default["cache_placement"]["network_cache"] = [0.5, 1, 5, 10]
NETWORK_CACHE = [1, 10]
default["cache_placement"]["name"] = "UNIFORM"

# Specify content placement
default["content_placement"]["name"] = "UNIFORM"

# List of all implemented topologies
# Topology implementations are located in ./icarus/scenarios/topology.py
TOPOLOGIES = [
    "GEANT",
    "WIDE",
    "GARR",
]

# Specify cache replacement policy
default["cache_policy"]["name"] = "QMARC"
# default["cache_policy"]["name"] = "LRU"
default["cache_policy"]["tiers"] = TIERS
default["cache_policy"]["alpha"] = 0.3

LATENCY_FUNCTION = {
      "high": {
        "thresholds": {
          "20.0": 0.0,
          "40.0": 5.0e-8,
          "60.0": 8.0e-8
        },
        "default": 1.0e-7
      },
      "low": {
        "thresholds": {
          "20.0": 0.0,
          "40.0": 2.0e-8,
          "60.0": 5.0e-8
        },
        "default": 8.0e-8
      }
    }

STRATEGIES = [
   "COST_CACHE",
    "PROB_CACHE",
    "CL4M",
    
    "LCE",
    
]

# Specify strategy params
strategy_params = {
    "COST_CACHE": {
        "cost_per_joule" : 0.020324,  # $/joule
        "cost_per_bit" : 1.2 * 10**-6,  # $/bit
        "router_energy_density" : 2 * 10**-8,  # j/bit
        "link_energy_density" : 1.5 * 10**-9,  # j/bit
        "latency_function": LATENCY_FUNCTION,
        "chunk_size" : 10 ** 5,
        "tiers" : TIERS,   
    },
}

DATA_COLLECTORS = {
    "COST": {
        "cost_params": strategy_params["COST_CACHE"],
        "tiers": TIERS
    },
    "CACHE_HIT_RATIO":{},
    "LATENCY": {},
}

# Create experiment configuration
EXPERIMENT_QUEUE = deque()
for strategy in STRATEGIES:
    for topology in TOPOLOGIES:
      for network_cache in NETWORK_CACHE:
        experiment = copy.deepcopy(default)
        experiment["strategy"]["name"] = strategy
        experiment["topology"]["name"] = topology
        experiment["cache_placement"]["network_cache"] = network_cache
        if strategy in strategy_params:
            experiment["strategy"].update(strategy_params[strategy])
        experiment[
            "desc"
        ] = "Strategy: {}, topology: {}, network cache: {}".format(
            strategy,
            topology, 
            str(network_cache),
        )
        EXPERIMENT_QUEUE.append(experiment)

