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
        
    @staticmethod
    def additive_price(tau, m, a):
        return m * tau + a

    def ic_price(self, tau, m, z, lambda_i_to_j, lambda_j_to_i):
        return m * tau + z * self.q_i_to_j(tau, lambda_i_to_j, lambda_j_to_i)

    @staticmethod
    def q_i_to_j(tau, lambda_i_to_j, lambda_j_to_i):
        alpha = lambda_i_to_j / (lambda_i_to_j + lambda_j_to_i)
        beta = lambda_i_to_j + lambda_j_to_i
        return alpha * (1 - np.exp(-beta * tau))

    def compute_earnings_components(self, sigma, w_func, lambda_i, trip_lengths):
        in_sigma = (trip_lengths >= sigma[0]) & (trip_lengths <= sigma[1])
        F_sigma = np.mean(in_sigma)
        if F_sigma == 0:
            return 0, float('inf'), 0
        W = np.mean([w_func(tau) for tau in trip_lengths[in_sigma]])
        T = 1 / (lambda_i * F_sigma) + np.mean(trip_lengths[in_sigma])
        R = W / T
        return W, T, R

    def compute_mu(self, sigma_1, sigma_2, trip_lengths):
        W_1, T_1, R_1 = self.compute_earnings_components(
            sigma_1,
            lambda tau: self.multiplicative_price(tau, self.R_1_target),
            self.lambda_1,
            trip_lengths
        )
        W_2, T_2, R_2 = self.compute_earnings_components(
            sigma_2,
            lambda tau: self.additive_price(tau, self.R_2_target, 0),
            self.lambda_2,
            trip_lengths
        )
        Q_1 = self.lambda_1_to_2 + self.lambda_1 * np.mean(
            [self.q_i_to_j(tau, self.lambda_1_to_2, self.lambda_2_to_1) for tau in trip_lengths]
        )
        Q_2 = self.lambda_2_to_1 + self.lambda_2 * np.mean(
            [self.q_i_to_j(tau, self.lambda_2_to_1, self.lambda_1_to_2) for tau in trip_lengths]
        )
        T_1_full = self.lambda_1 * np.mean((trip_lengths >= sigma_1[0]) & (trip_lengths <= sigma_1[1])) * T_1
        T_2_full = self.lambda_2 * np.mean((trip_lengths >= sigma_2[0]) & (trip_lengths <= sigma_2[1])) * T_2
        denom = T_1_full * Q_2 + T_2_full * Q_1
        mu_1 = (T_1_full * Q_2) / denom
        mu_2 = (T_2_full * Q_1) / denom
        return mu_1, mu_2, R_1, R_2    

 def total_earnings_rate(self, sigma_1, sigma_2, m_1, a_1, m_2, a_2, trip_lengths):
        w_1 = lambda tau: self.additive_price(tau, m_1, a_1)
        w_2 = lambda tau: self.additive_price(tau, m_2, a_2)
        mu_1, mu_2, _, _ = self.compute_mu(sigma_1, sigma_2, trip_lengths)
        _, _, R_1 = self.compute_earnings_components(sigma_1, w_1, self.lambda_1, trip_lengths)
        _, _, R_2 = self.compute_earnings_components(sigma_2, w_2, self.lambda_2, trip_lengths)
        return mu_1 * R_1 + mu_2 * R_2

    def optimize_additive_surge(self, trip_lengths, num_samples=10000):
        sigma_all = (0, float('inf'))  # Accept all trips

        # Non-surge state parameters: multiplicative pricing only
        m_1 = self.R_1_target
        a_1 = 0

        # Optimize additive component a_2 for surge state
        def earnings_diff(a_2):
            w_2 = lambda tau: self.additive_price(tau, self.R_2_target, a_2)
            _, _, R_2 = self.compute_earnings_components(sigma_all, w_2, self.lambda_2, trip_lengths)
            return (R_2 - self.R_2_target) ** 2

        result = minimize_scalar(earnings_diff, bounds=(0, 5), method='bounded')
        a_2 = result.x
        m_2 = self.R_2_target

        # Check incentive compatibility (IC) with different trip strategies
        strategies = {
            "All": sigma_all,
            "Short": (0, self.trip_mean),
            "Long": (self.trip_mean, float('inf'))
        }

        print("Non-surge (State 1): m_1 =", m_1, "a_1 =", a_1)
        w_1 = lambda tau: self.additive_price(tau, m_1, a_1)
        for name, sigma in strategies.items():
            _, _, R = self.compute_earnings_components(sigma, w_1, self.lambda_1, trip_lengths)
            print(f"Strategy {name}: R_1 = {R:.3f}")

        print("\nSurge (State 2): m_2 =", m_2, "a_2 =", a_2)
        w_2 = lambda tau: self.additive_price(tau, m_2, a_2)
        for name, sigma in strategies.items():
            _, _, R = self.compute_earnings_components(sigma, w_2, self.lambda_2, trip_lengths)
            print(f"Strategy {name}: R_2 = {R:.3f}")

        R_all = self.total_earnings_rate(sigma_all, sigma_all, m_1, a_1, m_2, a_2, trip_lengths)
        print(f"\nTotal Earnings Rate (accept all): {R_all:.3f}")

        return m_1, a_1, m_2, a_2
