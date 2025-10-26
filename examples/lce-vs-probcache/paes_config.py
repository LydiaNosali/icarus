from collections import deque
from config import *
import pickle
EXPERIMENT_QUEUE = deque()
EXPERIMENT_QUEUE.extend(pickle.load(open('exp.pkl','rb')))
