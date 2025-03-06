import numpy as np
import matplotlib.pyplot as plt
from trip_pricing_optimizer import TripPricingOptimizer


def main():
    np.random.seed(42)
    optimizer = TripPricingOptimizer()

    # Generate simulated trip lengths
    trip_lengths = optimizer.trip_length_dist(10000)

    # Optimize surge pricing parameters (additive surge)
    m_1, a_1, m_2, a_2 = optimizer.optimize_additive_surge(trip_lengths)

    tau = np.linspace(0, 1, 100)
    surge_mult = [optimizer.multiplicative_price(t, m_2) for t in tau]
    surge_add = [optimizer.additive_price(t, m_2, a_2) for t in tau]
    surge_ic = [optimizer.ic_price(t, m_2, 0.5, optimizer.lambda_2_to_1, optimizer.lambda_1_to_2) for t in tau]

    plt.plot(tau, surge_mult, label="Multiplicative (Surge)")
    plt.plot(tau, surge_add, label="Additive (Surge)")
    plt.plot(tau, surge_ic, label="IC Approx (Surge)")
    plt.xlabel("Trip Length (τ)")
    plt.ylabel("Payout ($)")
    plt.legend()
    plt.title("Pricing Functions Comparison")
    plt.show()


if __name__ == "__main__":
    main()
