"""This module contains all configuration information used to run simulations"""
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

# Number of times each experiment is replicated
N_REPLICATIONS = 1

# List of metrics to be measured in the experiments
# The implementation of data collectors are located in ./icarus/execution/collectors.py
DATA_COLLECTORS = ["CACHE_HIT_RATIO", "LATENCY", "COST"]

# Queue of experiments
EXPERIMENT_QUEUE = deque()
NETWORK_CACHE = 0.05

CACHES =  [
    {"name":"DRAM",
    "size_factor": 1 / 31
    }, 
    {"name":"SSD",
    "size_factor": 5 / 31 
    }, 
    {"name":"HDD",
    "size_factor": 25 / 31
    }]

COST_ALPHA = 0.3

# Create tree of experiment configuration
default = Tree()

# Create standard experiment configuration

# Specify workload
default["workload"] = {
    "name": "STATIONARY",
    "alpha": 0.8,
    "n_contents": 10 ** 5,
    "n_warmup": 10 ** 5,
    "n_measured": 4 * 10 ** 5,
    "rate": 1.0,
    "high_priority_rate" :0.2,
}

STRATEGIES = [
    "COST_CACHE",
    # "LCE",  # Leave Copy Everywhere
    # "NO_CACHE",  # No caching, shorest-path routing
    # "HR_SYMM",  # Symmetric hash-routing
    # "HR_ASYMM",  # Asymmetric hash-routing
    # "HR_MULTICAST",  # Multicast hash-routing
    # # "HR_HYBRID_AM",  # Hybrid Asymm-Multicast hash-routing
    # # "HR_HYBRID_SM",  # Hybrid Symm-Multicast hash-routing
    # # "CL4M",  # Cache less for more
    "PROB_CACHE",  # ProbCache
    # "LCD",  # Leave Copy Down
    # # "RAND_CHOICE",  # Random choice: cache in one random cache on path
    # # "RAND_BERNOULLI",  # Random Bernoulli: cache randomly in caches on path
    
]

# Specify cache placement
default["cache_placement"]["network_cache"] = NETWORK_CACHE
default["cache_placement"]["name"] = "UNIFORM"

# Specify content placement
default["content_placement"]["name"] = "UNIFORM"

# Specify cache replacement policy
# default["cache_policy"]["name"] = "ARC"
default["cache_policy"]["caches"] = CACHES
default["cache_policy"]["alpha"] = COST_ALPHA

# Specify topology
default["topology"]["name"] = "ROCKET_FUEL"
default["topology"]["asn"] = 1221
# default["strategy"]["name"] = "COST_CACHE"


# Create experiments multiplexing all desired parameters
# for strategy in ["LCE", "PROB_CACHE"]:
#     experiment = copy.deepcopy(default)
#     experiment["strategy"]["name"] = strategy
#     experiment["desc"] = "Strategy: %s" % strategy
#     EXPERIMENT_QUEUE.append(experiment)

for strategy in STRATEGIES:
    for cache in ["QMARC"]:
        experiment = copy.deepcopy(default)
        experiment["cache_policy"]["name"] = cache
        experiment["strategy"]["name"] = strategy
        experiment[
                    "desc"
                ] = "Cache Policy: {}, strategy: {}".format(
                   cache,
                    strategy,
                )
        EXPERIMENT_QUEUE.append(experiment)

