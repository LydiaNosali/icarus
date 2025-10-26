"""This module contains all configuration information used to run simulations"""
from collections import deque
import copy
from icarus.util import Tree

LOG_LEVEL = "INFO"
CACHING_GRANULARITY = "OBJECT"
RESULTS_FORMAT = "PICKLE"
PARALLEL_EXECUTION = False
N_REPLICATIONS = 1

TIERS = [
    {"name":"DRAM",
    "size_factor": 1/5,
    "purchase_cost" : 150, # in $
    "lifespan" : 5, # in years
    "read_throughput" : 4e+10,  # 40GBPS
    "write_throughput" : 2e+10, # 20GBPS
    "latency"  : 1e-7,  #100ns
    "active_caching_power_density" : 10**-9,  # w/bit
    "idle_power_density_per_bit" : 10**-12,  # w/bit
    "idle_power_density" : 0.1,  # w
    "embodied_kgco2e_per_gb":0.109, #kg CO2e per GB,
    },
    {"name":"SSD",
    "size_factor": 4/5,
    "purchase_cost" : 100, # in $
    "lifespan" : 3, # in years (SSD generally has a shorter lifespan compared to DRAM)
    "read_throughput" : 5e+9,  # 5GBPS (typically slower than DRAM)
    "write_throughput" : 2.5e+9, # 2.5GBPS (writing to SSD is slower than reading)
    "latency"  : 1e-5,  # 10 microseconds (latency is higher than DRAM)
    "active_caching_power_density" : 5e-7,  # 0.5 microwatts/bit (active power)
    "idle_power_density_per_bit" : 5e-9,  # 5 nanowatts/bit (idle power)
    "idle_power_density" : 0.2,  # w (idle power)
    "embodied_kgco2e_per_gb":0.6, #kg CO2e per GB,
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
        "cost_per_bit" : 1.2 * 10**-6,  # $/bit
        "router_energy_density" : 2 * 10**-8,  # j/bit
        "link_energy_density" : 1.5 * 10**-9,  # j/bit
        "penalty_table": PENALTY_TABLE,
        "chunk_size" : 10 ** 5,
    }
}

DATA_COLLECTORS = {
    "CACHE_HIT_RATIO": {},
    "LATENCY": {},
    "COST": {
        "cost_params": strategy_params["CL2SM"],
    },
   "CHRCP" : {},
    # "LINK_LOAD":{}
    # "REPLICA_MONITOR":{}
}

default = Tree()

default["workload"] = {
    "name": "STATIONARY",
    "n_contents": 10000,
    "n_warmup": 60000,
    "n_measured": 60000,
    "rate": 1,
    "high_priority_rate" :0.2,
    "priority_values": ["low", "high"],
    "data_size_range" : [1000, 8000],
    "seed": 1,
}

default["content_placement"]["name"] = "UNIFORM"
default["content_placement"]["seed"] = 1
default["cache_policy"]["name"] = "QMARC"
default["cache_policy"]["alpha"] = 0.3
default["cache_policy"]["tiers"] = TIERS
default["cache_placement"]["name"]="UNIFORM"

STRATEGIES = ["CL2SM", "LCE", "LCD", "PROB_CACHE", "CL4M", "CPCache"]
ALPHA = [0.8, 1.2, 2.0]

NETWORK_CACHE = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03] # which is 5% and 10%

TOPOLOGIES = [
    # "GEANT",
    "GARR",
    # "PATH",
    # "ROCKET_FUEL",
    # "WIDE",
    # "TISCALI",
]

# topology_params = {
#     "PATH": {
#         "n" : 4,
#     },
#     "ROCKET_FUEL" :{
#         "asn" :1221,
#     }
# }

EXPERIMENT_QUEUE = deque()


for topology in TOPOLOGIES:
    for alpha in ALPHA:
        for strategy in STRATEGIES:
            for network_cache in NETWORK_CACHE:
		experiment = copy.deepcopy(default)
		experiment["cache_placement"]["name"] = cache_placement
		experiment["cache_placement"]["network_cache"] = network_cache
		experiment["topology"]["name"] = topology
		experiment["strategy"]["name"] = strategy
		experiment["workload"]["alpha"] = alpha
		if strategy in strategy_params:
			experiment["strategy"].update(strategy_params[strategy])  
                    # if topology in topology_params:
                    #     experiment["topology"].update(topology_params[topology])
			experiment[
                        	"desc"
                    		] = "alpha: {}, strategy: {}, topology: {}, network cache: {}".format(
                        	str(alpha),
                        	strategy,
                        	topology,
                        	str(network_cache),
                    		)
                EXPERIMENT_QUEUE.append(experiment)

