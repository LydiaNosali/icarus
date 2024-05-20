"""This module contains all configuration information used to run simulations"""
from collections import deque
import copy
from icarus.util import Tree

# If True, executes simulations in parallel using multiple processes
# to take advantage of multicore CPUs
PARALLEL_EXECUTION = False

# Number of times each experiment is replicated
N_REPLICATIONS = 1

# List of metrics to be measured in the experiments
# The implementation of data collectors are located in ./icarus/execution/collectors.py
DATA_COLLECTORS = ["CACHE_HIT_RATIO", "LATENCY", "LINK_LOAD"]

# Queue of experiments
EXPERIMENT_QUEUE = deque()
NETWORK_CACHE = 0.01

CACHES =  [
    {"name":"DRAM",
     "size": NETWORK_CACHE / 31
     }, 
    {"name":"SSD",
     "size": NETWORK_CACHE * (5 / 31)
    }, 
    {"name":"HDD",
     "size": NETWORK_CACHE * (25 / 31)
    }
    ]

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
default["strategy"]["name"] = "COST_CACHE"

# Create experiments multiplexing all desired parameters
# for strategy in ["LCE", "PROB_CACHE"]:
#     experiment = copy.deepcopy(default)
#     experiment["strategy"]["name"] = strategy
#     experiment["desc"] = "Strategy: %s" % strategy
#     EXPERIMENT_QUEUE.append(experiment)
for cache in ["MARC", "QMARC"]:
    experiment = copy.deepcopy(default)
    experiment["cache_policy"]["name"] = cache
    experiment["desc"] = "Cache policy: %s" % cache
    EXPERIMENT_QUEUE.append(experiment)

