from collections import deque
import pickle
LOG_LEVEL = 'INFO'
CACHING_GRANULARITY = 'OBJECT'
RESULTS_FORMAT = 'PICKLE'
PARALLEL_EXECUTION = False
N_REPLICATIONS = 1
N_PERIODS = 1
EXPERIMENT_QUEUE = deque()
EXPERIMENT_QUEUE.extend(pickle.load(open('/Users/lydia/Desktop/icarus/icarus/scenarios/paes/paes.pkl', 'rb')))
