"""This module contains all configuration information used to run simulations"""
from multiprocessing import cpu_count
from collections import deque
import copy
from icarus.util import Tree

# # GENERAL SETTINGS

# # Level of logging output
# # Available options: DEBUG, INFO, WARNING, ERROR, CRITICAL
# LOG_LEVEL = "DEBUG"

# # If True, executes simulations in parallel using multiple processes
# # to take advantage of multicore CPUs
# PARALLEL_EXECUTION = False

# # Number of processes used to run simulations in parallel.
# # This option is ignored if PARALLEL_EXECUTION = False
# N_PROCESSES = cpu_count()

# # Granularity of caching.
# # Currently, only OBJECT is supported
# CACHING_GRANULARITY = "OBJECT"

# # Format in which results are saved.
# # Result readers and writers are located in module ./icarus/results/readwrite.py
# # Currently only PICKLE is supported
# RESULTS_FORMAT = "PICKLE"

# # Number of times each experiment is replicated
# N_REPLICATIONS = 1

# # List of metrics to be measured in the experiments
# # The implementation of data collectors are located in ./icarus/execution/collectors.py
# # DATA_COLLECTORS = ["CACHE_HIT_RATIO", "LATENCY", "COST"]


# TIERS = [
#     {"name":"DRAM",
#     "size_factor": 1 / 31,
#     "purchase_cost" : 150, # in $
#     "lifespan" : 5, # in years
#     "read_throughput" : 4e+10,  # 40GBPS
#     "write_throughput" : 2e+10, # 20GBPS
#     "latency"  : 1e-7,  #100ns
#     "active_caching_power_density" : 10**-9,  # w/bit
#     "idle_power_density" : 10**-9,  # w/bit
#     }, 
#     {"name":"SSD",
#     "size_factor": 5 / 31,
#     "purchase_cost" : 118, # in $
#     "lifespan" : 5, # in years
#     "read_throughput" : 3e+9,  # 3GBPS
#     "write_throughput" : 1e+9, # 1GBPS
#     "latency"  : 1e-5,  #10000ns
#     "active_caching_power_density" : 10**-9,  # w/bit
#     "idle_power_density" : 10**-9,  # w/bit
#     }, 
#     {"name":"HDD",
#     "size_factor": 25 / 31,
#     "purchase_cost" : 65, # in $
#     "lifespan" : 5, # in years
#     "read_throughput" : 1.5e+8,  #150MBPS
#     "write_throughput" : 3e+7, # 30MBPS
#     "latency"  : 4e-3,  #4ms
#     "active_caching_power_density" : 10**-9,  # w/bit
#     "idle_power_density" : 10**-9,  # w/bit
#     }]

# # Create tree of experiment configuration
# default = Tree()

# # Specify workload
# # default["workload"] = {
# #     "name": "STATIONARY",
# #     "alpha": 0.8,
# #     "n_contents": 6,
# #     "n_warmup": 3,
# #     "n_measured": 21,
# #     "rate": 1,
# #     "high_priority_rate" :0.2,
# #     "priority_values": ["low", "high"],
# #     "data_size_range" : [1024, 4096]
# # }
# default["workload"] = {
#     "name": "TRACE_DRIVEN",
#     "reqs_file": "/home/lydia/icarus/examples/lce-vs-probcache/events.csv",
#     "contents_file" : "/home/lydia/icarus/examples/lce-vs-probcache/eu_0.2_contents.csv",
#     "n_contents": 10,
#     "n_warmup": 10,
#     "n_measured": 10,
# }
# # Specify cache placement
# # default["cache_placement"]["network_cache"] = [0.5, 1, 5, 10]
# NETWORK_CACHE = [0.5, 0.8]
# default["cache_placement"]["name"] = "UNIFORM"

# # Specify content placement
# default["content_placement"]["name"] = "UNIFORM"

# # List of all implemented topologies
# # Topology implementations are located in ./icarus/scenarios/topology.py
# default["topology"]["name"] = "PATH"
# default["topology"]["n"] = 3
# # TOPOLOGY = {
# #     "PATH":{
# #        "n":3,
# #     },
# #     # "WIDE",
# #     # "GARR",
# # }

# # Specify cache replacement policy
# default["cache_policy"]["name"] = "QMARC"
# # default["cache_policy"]["name"] = "LRU"
# default["cache_policy"]["tiers"] = TIERS
# default["cache_policy"]["alpha"] = 0.3

# PENALTY_TABLE = [
#     {"delay": 20, "P0": 0.0, "P1": 0.0},        # Delay < 20 ms
#     {"delay": 60, "P0": 50, "P1": 10},     # Delay < 150 ms
#     {"delay": float('inf'), "P0": 75, "P1": 15}  # Delay >= 150 ms (use infinity for no upper limit)
# ]
# STRATEGIES = [
#    "LCE",
#     "COST_CACHE",
#     "CL4M",
#    "PROB_CACHE",
# ]

# # Specify strategy params
# strategy_params = {
#     "COST_CACHE": {
#         "cost_per_joule" : 0.020324,  # $/joule
#         "cost_per_bit" : 1.2 * 10**-6,  # $/bit
#         "router_energy_density" : 2 * 10**-8,  # j/bit
#         "link_energy_density" : 1.5 * 10**-9,  # j/bit
#         "penalty_table": PENALTY_TABLE,
#         "chunk_size" : 10 ** 5,
#         "tiers" : TIERS,   
#     },
# }

# DATA_COLLECTORS = {
#     "COST": {
#         "cost_params": strategy_params["COST_CACHE"],
#         "tiers": TIERS
#     },
#     "CACHE_HIT_RATIO":{},
#     "LATENCY": {},
# }

# # Create experiment configuration
# EXPERIMENT_QUEUE = deque()
# for strategy in STRATEGIES:
#     # for topology in TOPOLOGY:
#     for network_cache in NETWORK_CACHE:
#         experiment = copy.deepcopy(default)
#         experiment["strategy"]["name"] = strategy
#         # experiment["topology"]["name"] = topology
#         experiment["cache_placement"]["network_cache"] = network_cache
#         if strategy in strategy_params:
#             experiment["strategy"].update(strategy_params[strategy])
#         experiment[
#             "desc"
#         ] = "Strategy: {},network cache: {}".format(
#             strategy,
#             # topology, 
#             str(network_cache),
#         )
#         EXPERIMENT_QUEUE.append(experiment)

# GENERAL SETTINGS
LOG_LEVEL = "INFO"
PARALLEL_EXECUTION = False
N_REPLICATIONS = 1
CACHING_GRANULARITY = "OBJECT"
RESULTS_FORMAT = "PICKLE"

TIERS = [
    {"name":"DRAM",
    "size_factor": 1/31,
    "purchase_cost" : 150, # in $
    "lifespan" : 5, # in years
    "read_throughput" : 4e+10,  # 40GBPS
    "write_throughput" : 2e+10, # 20GBPS
    "latency"  : 1e-7,  #100ns
    "active_caching_power_density" : 10**-9,  # w/bit
    "idle_power_density" : 10**-12,  # w/bit
    },
    {"name":"SSD",
    "size_factor": 5/31,
    "purchase_cost" : 100, # in $
    "lifespan" : 3, # in years (SSD generally has a shorter lifespan compared to DRAM)
    "read_throughput" : 5e+9,  # 5GBPS (typically slower than DRAM)
    "write_throughput" : 2.5e+9, # 2.5GBPS (writing to SSD is slower than reading)
    "latency"  : 1e-5,  # 10 microseconds (latency is higher than DRAM)
    "active_caching_power_density" : 5e-7,  # 0.5 microwatts/bit (active power)
    "idle_power_density" : 5e-9,  # 5 nanowatts/bit (idle power)
    },
    {"name":"HDD",
    "size_factor": 15/31,
    "purchase_cost" : 50,  # in $ (cheaper than SSD and DRAM)
    "lifespan" : 3,  # in years (HDDs can vary, but a 3-year lifespan is a reasonable assumption)
    "read_throughput" : 1e+8,  # 100MB/s (0.1 GBPS, slower than SSD and DRAM)
    "write_throughput" : 5e+7,  # 50MB/s (slower than reads)
    "latency"  : 5e-3,  # 5 milliseconds (higher latency due to mechanical operations)
    "active_caching_power_density" : 8e-7,  # 0.8 μW/bit (active power)
    "idle_power_density" : 1e-8,  # 10 nanowatts/bit (idle power)
    }
]

STRATEGIES = ["Algo4", "LCE", "COST_CACHE", "CL4M", "PROB_CACHE"]
PENALTY_TABLE = [
    {"delay": 2, "P0": 0.0, "P1": 0.0},        # Delay < 20 ms
    {"delay": 6, "P0": 50, "P1": 10},     # Delay < 150 ms
    {"delay": float('inf'), "P0": 75, "P1": 15}  # Delay >= 150 ms (use infinity for no upper limit)
]
strategy_params = {
    "COST_CACHE": {
        "cost_per_joule" : 0.020324,  # $/joule
        "cost_per_bit" : 1.2 * 10**-6,  # $/bit
        "router_energy_density" : 2 * 10**-8,  # j/bit
        "link_energy_density" : 1.5 * 10**-9,  # j/bit
        "penalty_table": PENALTY_TABLE,
        "chunk_size" : 10 ** 5,
        "tiers" : TIERS,   
    },
     "Algo4": {
        "cost_per_joule" : 0.020324,  # $/joule
        "cost_per_bit" : 1.2 * 10**-6,  # $/bit
        "router_energy_density" : 2 * 10**-8,  # j/bit
        "link_energy_density" : 1.5 * 10**-9,  # j/bit
        "penalty_table": PENALTY_TABLE,
        "chunk_size" : 10 ** 5,
        "tiers" : TIERS,   
    },
}
DATA_COLLECTORS = {
    "CACHE_HIT_RATIO": {},
    "COST": {
        "cost_params": strategy_params["COST_CACHE"],
        "tiers": TIERS
    },
    "LATENCY": {},
}

NETWORK_CACHE = [0.1, 0.05] # which is 5% and 10%

default = Tree()
default["workload"] = {
    "name": "TRACE_DRIVEN",
    "reqs_file": "/home/lydia/icarus/examples/lce-vs-probcache/traces/events.csv",
    "contents_file" : "/home/lydia/icarus/examples/lce-vs-probcache/traces/events_contents.csv",
    "n_contents": 1000,
    "n_warmup": 1000,
    "n_measured": 2000,
}

# default["workload"] = {
#     "name": "STATIONARY",
#     "alpha": 0.8,
#     "n_contents": 1000,
#     "n_warmup": 1000,
#     "n_measured": 2000,
#     "rate": 1,
#     "high_priority_rate" :0.2,
#     "priority_values": ["low", "high"],
#     "data_size_range" : [1024, 4096]
# }
default["content_placement"]["name"] = "UNIFORM"
default["cache_placement"]["name"] = "UNIFORM"
default["cache_policy"]["name"] = "QMARC" 
default["cache_policy"]["tiers"] = TIERS
default["cache_policy"]["alpha"] = 0.3


default["topology"]["name"] = "PATH"
default["topology"]["n"] = 3

# Create experiment configuration
EXPERIMENT_QUEUE = deque()
for strategy in STRATEGIES:
    # for topology in TOPOLOGY:
    for network_cache in NETWORK_CACHE:
        experiment = copy.deepcopy(default)
        experiment["strategy"]["name"] = strategy
        # experiment["topology"]["name"] = topology
        experiment["cache_placement"]["network_cache"] = network_cache
        if strategy in strategy_params:
            experiment["strategy"].update(strategy_params[strategy])
        experiment[
            "desc"
        ] = "Strategy: {},network cache: {}".format(
            strategy,
            # topology, 
            str(network_cache),
        )
        EXPERIMENT_QUEUE.append(experiment)