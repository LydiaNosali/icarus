"""This module contains all configuration information used to run simulations"""
from multiprocessing import cpu_count
from collections import deque
import copy
from icarus.util import Tree


# GENERAL SETTINGS
LOG_LEVEL = "INFO"
PARALLEL_EXECUTION = True
N_REPLICATIONS = 2
CACHING_GRANULARITY = "OBJECT"
RESULTS_FORMAT = "PICKLE"

TIERS = [
    {"name":"DRAM",
    "size_factor": 1/5,
    "purchase_cost" : 150, # in $
    "lifespan" : 5, # in years
    "read_throughput" : 4e+10,  # 40GBPS
    "write_throughput" : 2e+10, # 20GBPS
    "latency"  : 1e-7,  #100ns
    "active_caching_power_density" : 10**-9,  # w/bit
    "idle_power_density" : 10**-12,  # w/bit
    },
    {"name":"SSD",
    "size_factor": 4/5,
    "purchase_cost" : 100, # in $
    "lifespan" : 3, # in years (SSD generally has a shorter lifespan compared to DRAM)
    "read_throughput" : 5e+9,  # 5GBPS (typically slower than DRAM)
    "write_throughput" : 2.5e+9, # 2.5GBPS (writing to SSD is slower than reading)
    "latency"  : 1e-5,  # 10 microseconds (latency is higher than DRAM)
    "active_caching_power_density" : 5e-7,  # 0.5 microwatts/bit (active power)
    "idle_power_density" : 5e-9,  # 5 nanowatts/bit (idle power)
    }
]

STRATEGIES = [ "COST", "LCD","RAND_CHOICE", "LCE", "PROB_CACHE"]
# POLICIES = ["QMARC"]
# STRATEGIES = ["COST"]
POLICIES = ["QMARC", "QMARC", "QMARC", "QMARC", "QMARC"]


PENALTY_TABLE = [
    {"delay": 2, "P0": 0.0, "P1": 0.0},        # Delay < 20 ms
    {"delay": 6, "P0": 50, "P1": 10},     # Delay < 150 ms
    {"delay": float('inf'), "P0": 75, "P1": 15}  # Delay >= 150 ms (use infinity for no upper limit)
]
strategy_params = {
    "COST": {
        "cost_per_joule" : 0.020324,  # $/joule
        "cost_per_bit" : 1.2 * 10**-6,  # $/bit
        "router_energy_density" : 2 * 10**-8,  # j/bit
        "link_energy_density" : 1.5 * 10**-9,  # j/bit
        "penalty_table": PENALTY_TABLE,
        "chunk_size" : 10 ** 5,
        "tiers" : TIERS,   
    }
}
DATA_COLLECTORS = {
    "CACHE_HIT_RATIO": {},
    "COST": {
        "cost_params": strategy_params["COST"],
        "tiers": TIERS
    },
    "LATENCY": {},
    "CHRCP" : {}
}

# NETWORK_CACHE = [0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.5, 0.8, 1.0] # which is 5% and 10%
NETWORK_CACHE = [0.2, 0.3] # which is 5% and 10%

# NETWORK_CACHE = [0.1] # 10%
# NETWORK_CACHE = [0.01] # 1%
# NETWORK_CACHE = [0.001] # 0.1%
# NETWORK_CACHE = [0.0005] # 0.05%


default = Tree()
default["workload"] = {
    "name": "TRACE_DRIVEN",
    "reqs_file": "/home/lydia/icarus/examples/lce-vs-probcache/overfitting/events.csv",
    "contents_file" : "/home/lydia/icarus/examples/lce-vs-probcache/overfitting/events_contents.csv",
    "n_contents": 768,
    "n_warmup": 1000,
    "n_measured": 2000,
}

# default["workload"] = {
#     "name": "STATIONARY",
#     "alpha": 1.2,
#     "n_contents": 3 * 10 ** 3,
#     "n_warmup": 3 * 10 ** 3,
#     "n_measured": 6 * 10 ** 3,
#     "rate": 1,
#     "high_priority_rate" :0.2,
#     "priority_values": ["low", "high"],
#     "data_size_range" : [1000, 8000],
#     "seed" : 1
# }

default["content_placement"]["name"] = "UNIFORM"
default["cache_placement"]["name"] = "UNIFORM"
# default["cache_policy"]["name"] = "QMARC"
default["cache_policy"]["tiers"] = TIERS
default["cache_policy"]["alpha"] = 0.3

# default["topology"]["name"] = "WIDE"
TOPOLOGIES = [
    # "GEANT",
    # "WIDE",
    # "GARR",
    "TISCALI",
]
# default["topology"]["name"] = "PATH"
# default["topology"]["n"] = 3

# Create experiment configuration
EXPERIMENT_QUEUE = deque()
for strategy, policy in zip(STRATEGIES, POLICIES):
    for network_cache in NETWORK_CACHE:
        for topology in TOPOLOGIES:
            experiment = copy.deepcopy(default)
            experiment["strategy"]["name"] = strategy
            experiment["cache_policy"]["name"] = policy  # Include the policy corresponding to the strategy
            experiment["topology"]["name"] = topology
            experiment["cache_placement"]["network_cache"] = network_cache
            if strategy in strategy_params:
                experiment["strategy"].update(strategy_params[strategy])
            experiment[
                "desc"
            ] = "Strategy: {}, Policy: {}, Topology: {}, Network cache: {}".format(
                strategy,
                policy,
                topology, 
                str(network_cache),
            )
            EXPERIMENT_QUEUE.append(experiment)