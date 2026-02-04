"""This module contains all configuration information used to run simulations"""
from collections import deque
import copy
from icarus.util import Tree

LOG_LEVEL = "INFO"
CACHING_GRANULARITY = "OBJECT"
RESULTS_FORMAT = "PICKLE"
PARALLEL_EXECUTION = False
N_REPLICATIONS = 1
N_PERIODS = 24

TIERS = [ 
    {
        "name":"DRAM", 
        "size_factor": 0.2, 
        "purchase_cost" : 5e-9, # $/Byte
        "lifespan" : 3, # in years 
        "read_throughput" : 4e+10, # 40GBPS 
        "write_throughput" : 2e+10, # 20GBPS 
        "latency" : 1e-7, #100ns 
        "active_caching_power_density" : 1e-6, # w/bit 
        "idle_power_density_per_bit" : 1e-8, # w/bit 
        "embodied_kgco2e_per_gb":0.4, #kg CO2e per GB 
    }, 
    {
        "name":"SSD", 
        "size_factor": 0.3, 
        "purchase_cost" : 9e-10, # $/Byte
        "lifespan" : 2, # in years (SSD generally has a shorter lifespan compared to DRAM) 
        "read_throughput" : 5e+9, # 5GBPS (typically slower than DRAM)
        "write_throughput" : 2.5e+9, # 2.5GBPS (writing to SSD is slower than reading) 
        "latency" : 1e-5, # 10 microseconds (latency is higher than DRAM) 
        "active_caching_power_density" : 2e-7, # 0.5 microwatts/bit (active power) 
        "idle_power_density_per_bit" : 1e-9, # 5 nanowatts/bit (idle power) 
        "embodied_kgco2e_per_gb":1.8, #kg CO2e per GB
        },
    {
        "name": "HDD",
        "size_factor": 0.5,                  # 💾 Largest tier (~60%)
        "purchase_cost": 1.86e-11,                # $/Byte
        "lifespan": 5,                       # years
        "read_throughput": 2e8,              # 200 MB/s
        "write_throughput": 1.5e8,           # 150 MB/s
        "latency": 3e-3,                     # 5 ms
        "active_caching_power_density": 5e-8,   # W/bit  (~0.4 W/GB)
        "idle_power_density_per_bit": 8e-9,     # W/bit  (~0.06 W/GB)
        "embodied_kgco2e_per_gb": 0.25,      # low embodied carbon per GB
        }
    ]

PENALTY_TABLE = [
    {"delay": 50, "P0": 0.0, "P1": 0.0},        # Delay < 20 ms
    {"delay": 80, "P0": 50.0, "P1": 10.0},     # Delay < 150 ms
    {"delay": float('inf'), "P0": 75.0, "P1": 35.0}  # Delay >= 150 ms (use infinity for no upper limit)
]

strategy_params = {
    "CL2SM": {
        "cost_per_joule" : 0.020324,  # $/joule
        "cost_per_bit" : 1.2e-6,  # $/bit
        "router_energy_density" : 2e-8,  # j/bit
        "link_energy_density" : 1.5e-9,  # j/bit
        "penalty_table": PENALTY_TABLE,
        "chunk_size" : 10e5,
    }
}

DATA_COLLECTORS = {
    "CACHE_HIT_RATIO": {},
    "LATENCY": {},
    "COST": {
        "cost_params": strategy_params["CL2SM"],
    },
    "CARBONFOOTPRINT":{
        "cost_params": strategy_params["CL2SM"],
        "tiers": TIERS,
    },
    "CCHRP":{},
    # "CHRCP" : {},
    # "LINK_LOAD":{}
    # "REPLICA_MONITOR":{}
}

default = Tree()

default["workload"] = {
    "name": "STATIONARY",
    "n_contents": 300000,
    "n_warmup": 600000,
    "n_measured": 600000 / N_PERIODS,
    "rate": 12,
    "high_priority_rate" :0.2,
    "priority_values": ["low", "high"],
    "data_size_range" : [1000, 8000],
    "seed": 1,
}
    # "name": "STATIONARY",
    # "n_contents": 10000,
    # "n_warmup": 50000,
    # "n_measured": 50000,
    # "rate": 5,
    # "high_priority_rate" :0.2,
    # "priority_values": ["low", "high"],
    # "data_size_range" : [1000, 8000],
    # "seed": 1,

default["content_placement"]["name"] = "UNIFORM"
default["content_placement"]["seed"] = 1
default["cache_policy"]["name"] = "QMARC"
default["cache_policy"]["alpha"] = 0.3
default["cache_policy"]["tiers"] = TIERS

# CACHE_PLACEMENT = ["ALLOCATED", "HYBRID_GREEN_CENTRALITY","GREEN", "BETWEENNESS_CENTRALITY", "UNIFORM", "DEGREE", "RANDOM","OPTIMAL_MEDIAN", "OPTIMAL_HASHROUTING"]
# CACHE_PLACEMENT = ["SCORE_BASED", "UNIFORM", "BETWEENNESS_CENTRALITY"]
# CACHE_PLACEMENT = ["UNIFORM", "BETWEENNESS_CENTRALITY"]
# CACHE_PLACEMENT = ["UNIFORM"]
# CACHE_PLACEMENT = ["BETWEENNESS_CENTRALITY"]
# CACHE_PLACEMENT = ["GREEN"]
# CACHE_PLACEMENT = ["GREEN", "EIGENVECTOR_CENTRALITY", "CACHECRAFT", "BETWEENNESS_CENTRALITY", "UNIFORM", "DEGREE"]
# CACHE_PLACEMENT = ["GREEN", "CACHECRAFT", "BETWEENNESS_CENTRALITY", "UNIFORM", "DEGREE"]
# CACHE_PLACEMENT = ["CACHECRAFT", "BETWEENNESS_CENTRALITY", "UNIFORM", "DEGREE"]
CACHE_PLACEMENT = ["GREEN"]

# STRATEGIES = ["CL2SM", "LCE", "LCD", "PROB_CACHE", "CL4M", "CPCache"]
STRATEGIES = ["CL2SM"]
# ALPHA = [0.8, 1.2, 2.0]
# ALPHA = [0.6, 0.8, 1.2]
ALPHA = [0.8]

# NETWORK_CACHE = [0.01, 0.015, 0.02, 0.05] 
# NETWORK_CACHE = [0.001, 0.005, 0.01, 0.015, 0.02]
NETWORK_CACHE = [0.05]
# NETWORK_CACHE = [0.03, 0.05, 0.1]
# NETWORK_CACHE = [0.015, 0.02]
# NETWORK_CACHE = [0.6] 
print(NETWORK_CACHE)

TOPOLOGIES = [
    # "CARBON"
    "GEANT",
    # "GARR",
    # "PATH",
    # "ROCKET_FUEL",
    # "WIDE",
    # "TISCALI",
    # "TELEKOM",
]

topology_params = {
    "PATH": {
        "n" : 4,
    },
    "ROCKET_FUEL" :{
        "asn" :1221,
    }
}

cache_placement_params = {
    "OPTIMAL_MEDIAN" :{
        "n_cache_nodes" :13,
        "hit_ratio" : 1.0
    },
    "RANDOM": {
        "n_cache_nodes" : 13
    },
    "OPTIMAL_HASHROUTING" :{
        "n_cache_nodes" :13,
        "hit_ratio" : 1.0
    },
    "GREEN" :{
        "seed": 1.0,
        "RGN" : 1.0,
        "MAX_EVALUATION" :200,
    },
    "HYBRID_GREEN_CENTRALITY":{
        "seed": 1.0,
        "alpha": 0.0,
    },
    "ALLOCATED": {"allocations":[]},
    "CACHECRAFT": {
        "alpha": 0.85,
        "max_iter": 100,
        "tol": 1.0e-6
    },
    "EIGENVECTOR_CENTRALITY": {
        "max_iter": 1000,
        "tol": 1.0e-6
    }
}

EXPERIMENT_QUEUE = deque()


for topology in TOPOLOGIES:
    for alpha in ALPHA:
        for strategy in STRATEGIES:
            for cache_placement in CACHE_PLACEMENT:
                for network_cache in NETWORK_CACHE:
                    experiment = copy.deepcopy(default)
                    experiment["cache_placement"]["name"] = cache_placement
                    experiment["cache_placement"]["network_cache"] = network_cache
                    experiment["topology"]["name"] = topology
                    experiment["strategy"]["name"] = strategy
                    experiment["workload"]["alpha"] = alpha
                    if strategy in strategy_params:
                        experiment["strategy"].update(strategy_params[strategy])  
                    if cache_placement in cache_placement_params:
                        experiment["cache_placement"].update(cache_placement_params[cache_placement])
                    if topology in topology_params:
                        experiment["topology"].update(topology_params[topology])
                    experiment[
                        "desc"
                    ] = "alpha: {}, strategy: {}, topology: {}, network cache: {}".format(
                        str(alpha),
                        strategy,
                        topology,
                        str(network_cache),
                    )
                    EXPERIMENT_QUEUE.append(experiment)
