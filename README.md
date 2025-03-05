# Uber Surge Pricing

This project reproduces Uber's surge pricing mechanisms—multiplicative, additive, and incentive-compatible (IC). It simulates a dynamic ride-hailing environment with surge and non-surge states, computes driver earnings, and optimizes pricing to ensure drivers accept all trips (IC), aligning with Uber's evolution from multiplicative to additive surge pricing.

## Overview

Surge pricing balances supply and demand in ride-hailing platforms. This implementation models:
1. **Multiplicative Surge**: Fares scale with trip length (e.g., `w(tau) = m * tau`), used historically by Uber (Section 1).
2. **Additive Surge**: Adds a flat bonus to base fares (e.g., `w(tau) = m * tau + a`), Uber’s updated approach (Section 5).
3. **Incentive-Compatible (IC) Surge**: Approximates the theoretical form `w_i(tau) = m_i * tau + z_i * q_i_to_j(tau)` (Theorem 3), ensuring drivers maximize earnings by accepting all trips.

The simulation uses a two-state Continuous-Time Markov Chain (CTMC) for surge dynamics and a Poisson process for trip arrivals, reflecting the paper’s stochastic model (Section 2).

**Note**: Formulas are written as code (e.g., `w(tau) = m * tau`) for readability in plain Markdown. For LaTeX rendering (e.g., \( w(\tau) = m \tau \)), view in a LaTeX-enabled environment like GitHub Pages with extensions.

## Features
- Simulates surge (state 2) and non-surge (state 1) periods with transition rates `lambda_1_to_2 = 1`, `lambda_2_to_1 = 4`.
- Models trip lengths with a Weibull distribution (shape=2, mean=1/3), per Section 4.2.
- Computes driver earnings `R(w, sigma) = mu_1 * R_1 + mu_2 * R_2` (Lemma 1).
- Optimizes additive surge to hit target earnings (`R_1 = 2/3`, `R_2 = 1`) and ensure IC.
- Visualizes pricing functions to compare payout structures (Figure 2a-inspired).

