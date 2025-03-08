import numpy as np
from scipy.stats import weibull_min
from scipy.optimize import minimize_scalar


class TripPricingOptimizer:
    def __init__(self,
                 lambda_1=12,
                 lambda_2=12,
                 lambda_1_to_2=1,
                 lambda_2_to_1=4,
                 trip_mean=1/3,
                 R_1_target=2/3,
                 R_2_target=1):
        self.lambda_1 = lambda_1          # Non-surge trip arrival rate
        self.lambda_2 = lambda_2          # Surge trip arrival rate
        self.lambda_1_to_2 = lambda_1_to_2  # Transition rate from non-surge to surge
        self.lambda_2_to_1 = lambda_2_to_1  # Transition rate from surge to non-surge
        self.trip_mean = trip_mean        # Mean trip length (Weibull mean)
        self.R_1_target = R_1_target      # Target earnings rate in non-surge
        self.R_2_target = R_2_target      # Target earnings rate in surge
        self.weibull_shape = 2      

    def trip_length_dist(self, size):
        """
        Generate trip lengths using a Weibull distribution.
        Scale is adjusted so that the mean equals self.trip_mean.
        """
        scale = self.trip_mean / np.sqrt(np.pi / 2)
        return weibull_min.rvs(self.weibull_shape, scale=scale, size=size)

    @staticmethod
    def multiplicative_price(tau, m):
        return m * tau
     