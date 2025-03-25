import numpy as np
from scipy.optimize import linear_sum_assignment

class CombinatorialOptimizer:
    def __init__(self):
        pass
    
    def assign_orders(self, orders, drivers):
        n_orders = len(orders)
        n_drivers = len(drivers)
        cost_matrix = np.zeros((n_orders, n_drivers))

        # Build cost matrix with negative pickup distances
        for i, order in enumerate(orders):
            origin = order[0]
            for j, driver in enumerate(drivers):
                driver_loc = driver[0]
                pickup_dist = compute_distance(origin, driver_loc)
                cost_matrix[i, j] = -pickup_dist  

        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignments = [(i, col_ind[i]) for i in row_ind if col_ind[i] < n_drivers]
        total_cost = cost_matrix[row_ind, col_ind].sum()
        return assignments, -total_cost  # Return positive total distance
