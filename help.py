import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.optim as optim

def grid_to_index(loc):
    #Convert 2D grid coordinates to a flat index for a 5x5 grid.
    x, y = loc
    return int(x * 5 + y)

def compute_distance(loc1, loc2):
    return np.linalg.norm(np.array(loc1) - np.array(loc2))
