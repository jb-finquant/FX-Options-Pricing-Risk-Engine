# FX Options Pricing & Risk Engine

A quantitative finance project implementing an FX options pricing and risk management engine in Python. Built to demonstrate derivatives pricing, Greeks, risk modelling, delta hedging, and stochastic volatility — using EUR/USD market data from Yahoo Finance.

---

## Motivation

My background is in corporate finance, with experience in M&A and private equity. Alongside this, I developed a strong interest in financial markets and quantitative methods through my Bachelor's thesis, where I worked on the predictive power of the Variance Risk Premium and future stock returns.

This project was built to deepen my understanding of derivative pricing and risk management by translating financial theory into code. Rather than treating pricing models as black boxes, the objective was to understand their assumptions, implementation, and practical limitations.

The framework starts with the analytical Black-Scholes / Garman-Kohlhagen model for FX options and uses it as a benchmark for Monte Carlo pricing. It is then extended to Greeks, Value-at-Risk, stochastic volatility modelling, and discrete delta hedging to explore how model performance changes once some of the simplifying assumptions of Black-Scholes are relaxed.

Ultimately, the goal of the project is not only to price derivatives, but also to understand the trade-offs between mathematical models and real-world market behaviour.

---

## What This Project Covers

**Pricing**
- Garman-Kohlhagen closed-form solution for European FX options
- Monte Carlo pricing via Geometric Brownian Motion with Antithetic Variates
- Convergence analysis: MC pricing converging to BS benchmark as N → ∞

![MC Convergence to Black-Scholes](docs/images/mc_convergence.png)
![Simulated GBM Paths](docs/images/gbm_simulated_paths.png)

**Greeks**
- Analytical Greeks (Delta, Gamma, Vega, Theta, Rho) via Garman-Kohlhagen
- Numerical Greeks via central finite differences (Monte Carlo)
- Visualisation of all Greeks across a range of spot rates

![Greeks Across Spot Rates](docs/images/greeks_vs_spot.png)

**Risk**
- Monte Carlo VaR, Parametric VaR, Historical VaR
- P&L distribution over a one-day risk horizon
- Real EUR/USD market data via yfinance

![P&L Distribution with VaR](docs/images/pnl_var_distribution.png)

**Delta Hedging**
- Discrete delta hedging simulation for short and long call positions
- P&L attribution: option P&L vs hedge P&L vs hedging error
- Theta/Gamma decomposition of the hedging error

![Delta Hedging](docs/images/delta_hedging.png)

**Stochastic Volatility**
- Euler-Maruyance simulation of spot and variance paths
- Heston vs BS price comparison
- Emergent volatility smile from Heston MC prices via implied vol extraction
- Sensitivity analysis: smile shape as a function of ρ and ξ

![Heston Model](docs/images/heston_model.png)
![Heston Sensitivity](docs/images/heston_sensitivity.png)

**Implied Volatility**
- Brent's root-finding method to invert the BS pricing formula
- Used to extract implied vols from Heston-generated market prices

---

## Project Structure

```
├── main.py                  # Pricing, Greeks, VaR, Convergence
├── run_hedge.py             # Delta Hedging Simulation
├── run_heston.py            # Heston Model & Volatility Smile
├── config.py                # Global seed
│
├── model/
│   └── parameters.py        # FXModelParameters dataclass
├── data/
│   └── market_data.py       # EUR/USD data via yfinance
├── simulation/
│   └── gbm.py               # GBM with Antithetic Variates
├── pricing/
│   ├── black_scholes.py     # Garman-Kohlhagen analytical pricing
│   └── monte_carlo.py       # MC pricing
├── greeks/
│   └── finite_differences.py # Analytical & numerical Greeks
├── risk/
│   └── var.py               # MC, Parametric, Historical VaR
├── volatility/
│   ├── heston.py            # Heston simulation, pricing, smile
│   └── implied_vol.py       # Implied vol via Brent's method
└── hedging/
    └── delta_hedge.py       # Delta hedging & P&L attribution
```

---

## Getting Started

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the pricing engine:**
```bash
python main.py
```

**Run delta hedging simulation:**
```bash
python run_hedge.py
```

**Run Heston model:**
```bash
python run_heston.py
```

---

## Key Results

- MC pricing converges to BS benchmark with error scaling at 1/√N
- Delta hedging error decomposed into Theta gains and Gamma costs
- Heston model generates an emergent volatility smile — absent under BS constant volatility assumption
- Negative spot-vol correlation (ρ) produces volatility skew; symmetric smile emerges as ρ → 0

---

## References

- Garman, M.B. & Kohlhagen, S.W. (1983). *Foreign Currency Option Values.* Journal of International Money and Finance, 2(3), 231–237.
- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy, 81(3), 637–654.
- Heston, S.L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility.* Review of Financial Studies, 6(2), 327–343.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering.* Springer.
- Hull, J. (2022). *Options, Futures, and Other Derivatives.* Pearson.
- Brent, R.P. (1973). *Algorithms for Minimization without Derivatives.* Prentice-Hall.
- Gatheral, J. (2006). *The Volatility Surface.* Wiley.