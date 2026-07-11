import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style

from config import SEED
from model.parameters import FXModelParameters
from data.market_data import load_market_data
from simulation.gbm import run_simulation
from pricing.monte_carlo import compute_price
from pricing.black_scholes import black_scholes_fx_call, black_scholes_fx_put
from greeks.finite_differences import mc_delta, mc_gamma, mc_vega, mc_theta, mc_rho, bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho, greek_over_spot
from risk.var import compute_mc_var, compute_parametric_var, compute_historical_var

# --- Market Data ---
S0, vol, returns, date_from, date_to = load_market_data()

print(Fore.CYAN + "=" * 70 + Style.RESET_ALL)
print("MARKET DATA")
print(f"{'Period:':<20} {date_from} to {date_to}")
print(Fore.LIGHTWHITE_EX + "~" * 45 + Style.RESET_ALL)
print(f"{'Ticker:':<20} EURUSD=X")
print(f"{'Spot (S0):':<20} {S0:.4f}")
print(f"{'Realized Vol:':<20} {vol:.4f}")
print(f"{'Historical Returns:':<20} {len(returns)} days")

# --- Parameters & Pricing ---
params = FXModelParameters(S0=S0, vol=vol, K=round(S0, 4))

paths = run_simulation(params, seed=SEED)
mc_price = compute_price(params, paths)
bs_price = (black_scholes_fx_call(S0=params.S0, K=params.K, rd=params.domestic_r,
                          rf=params.foreign_r, vol=params.vol, T=params.T)
    if params.OptionsType == "Call"
    else black_scholes_fx_put(S0=params.S0, K=params.K, rd=params.domestic_r,
                              rf=params.foreign_r, vol=params.vol, T=params.T))

print(Fore.CYAN + "=" * 70 + Style.RESET_ALL)
print(f"{'MC Price:':<20} {mc_price: .6f}")
print(f"{'BS Price:':<20} {bs_price: .6f}")
print(Fore.GREEN + f"{'Difference:':<20} {mc_price - bs_price: .6f}" + Style.RESET_ALL)

# --- Greeks ---
greeks = {
    "Delta": (mc_delta(params), bs_delta(params)),
    "Gamma": (mc_gamma(params), bs_gamma(params)),
    "Vega":  (mc_vega(params),  bs_vega(params)),
    "Theta": (mc_theta(params), bs_theta(params)),
    "Rho":   (mc_rho(params),   bs_rho(params)),
}

print(Fore.CYAN + "=" * 70 + Style.RESET_ALL)
for name, (mc, bs) in greeks.items():
    print(f"{name:<10} MC: {mc: .6f}   BS: {bs: .6f}   Error: {mc-bs: .6f}")
print(Fore.CYAN + "=" * 70 + Style.RESET_ALL)

# --- VaR ---
var_mc, pnl = compute_mc_var(params, mc_price)

var_results = {
    "MC VaR":         var_mc,
    "Parametric VaR": compute_parametric_var(pnl, params.confidence_level),
    "Historical VaR": compute_historical_var(returns, params.confidence_level),
}

for name, value in var_results.items():
    print(f"{name:<20} {value:.6f}")
print(Fore.CYAN + "=" * 70 + Style.RESET_ALL)

# --- Convergence ---
def price_for_n(n: int, params: FXModelParameters) -> float:
    params_n = FXModelParameters(**{**params.__dict__, 'MonteCarloSimulations': n})
    return compute_price(params_n, run_simulation(params_n, seed=SEED))

sim_range = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000]
mc_prices = [price_for_n(n, params) for n in sim_range]

print()
print("CONVERGENCE ANALYSIS")
print(Fore.LIGHTWHITE_EX + "~" * 30 + Style.RESET_ALL)
print(f"{'BS Benchmark:':<20} {bs_price:.6f}")
print(f"{'Final MC Price:':<20} {mc_prices[-1]:.6f}")
print(Fore.GREEN + f"{'Final Error:':<20} {mc_prices[-1] - bs_price:.6f}" + Style.RESET_ALL)
print()
print(Fore.RED + "As N increases, the Monte Carlo pricing converges to the Black-Scholes benchmark.")
print("The error scales with 1/sqrt(N) – halving with every 4x increase in simulations." + Style.RESET_ALL)
print(Fore.CYAN + "=" * 70 + Style.RESET_ALL)

# --- Visualizations ---
fig, ax = plt.subplots()
ax.hist(pnl, bins=50, color="blue", alpha=0.7, label="P&L distribution")
ax.axvline(x=var_results["MC VaR"],         color="red",    linestyle="--", label="MC VaR 95%")
ax.axvline(x=var_results["Parametric VaR"], color="orange", linestyle="--", label="Parametric VaR 95%")
ax.axvline(x=var_results["Historical VaR"], color="green",  linestyle="--", label="Historical VaR 95%")
ax.set_title("P&L distribution")
ax.set_xlabel("P&L")
ax.set_ylabel("Frequency")
ax.legend()
plt.savefig("docs/images/pnl_var_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

spot_range = np.linspace(0.90, 1.30, 100)
prices = np.array([
    black_scholes_fx_call(S0=s, K=params.K, rd=params.domestic_r,
                          rf=params.foreign_r, vol=params.vol, T=params.T)
    for s in spot_range
])

deltas = greek_over_spot(bs_delta, params, spot_range)
gammas = greek_over_spot(bs_gamma, params, spot_range)
vegas  = greek_over_spot(bs_vega,  params, spot_range)
thetas = greek_over_spot(bs_theta, params, spot_range)
rhos   = greek_over_spot(bs_rho,   params, spot_range)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 8))
ax1.plot(spot_range, prices, color="blue",   label="Option Price")
ax1.set_title("Change of Price for different Spot Rates")
ax1.set_xlabel("Spot Rate")
ax1.set_ylabel("Option Price")
ax1.legend()

ax2.plot(spot_range, deltas, color="orange", label="Delta")
ax2.set_title("Delta for different Spot Rates")
ax2.set_xlabel("Spot Rate")
ax2.set_ylabel("Delta")
ax2.legend()

ax3.plot(spot_range, gammas, color="brown",  label="Gamma")
ax3.set_title("Gamma for different Spot Rates")
ax3.set_xlabel("Spot Rate")
ax3.set_ylabel("Gamma")
ax3.legend()

ax4.plot(spot_range, vegas,  color="green",  label="Vega")
ax4.set_title("Vega for different Spot Rates")
ax4.set_xlabel("Spot Rate")
ax4.set_ylabel("Vega")
ax4.legend()

ax5.plot(spot_range, thetas, color="red",    label="Theta")
ax5.set_title("Theta for different Spot Rates")
ax5.set_xlabel("Spot Rate")
ax5.set_ylabel("Theta")
ax5.legend()

ax6.plot(spot_range, rhos,   color="purple", label="Rho")
ax6.set_title("Rho for different Spot Rates")
ax6.set_xlabel("Spot Rate")
ax6.set_ylabel("Rho")
ax6.legend()

plt.tight_layout()
plt.savefig("docs/images/greeks_vs_spot.png", dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(paths[:50, :].T, alpha=0.4, linewidth=0.8)
ax.axhline(y=params.S0, color="black", linestyle="--", linewidth=1.5, label=f"S0 = {params.S0:.4f}")
ax.axhline(y=params.K,  color="red",   linestyle="--", linewidth=1.5, label=f"Strike = {params.K}")
ax.set_title("Simulated GBM Paths")
ax.set_xlabel("Time Steps (Days)")
ax.set_ylabel("Spot Rate")
ax.legend()
plt.tight_layout()
plt.savefig("docs/images/gbm_simulated_paths.png", dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sim_range, mc_prices, color="blue", marker="o", label="MC Price")
ax.axhline(y=bs_price, color="red", linestyle="--", label=f"BS Price = {bs_price:.5f}")
ax.set_title("MC convergence to Black-Scholes")
ax.set_xlabel("Number of Simulations")
ax.set_ylabel("Option Price")
ax.legend()
ax.set_xscale("log")
plt.tight_layout()
plt.savefig("docs/images/mc_convergence.png", dpi=150, bbox_inches="tight")
plt.show()