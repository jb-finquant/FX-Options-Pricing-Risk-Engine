import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style

from config import SEED
from model.parameters import FXModelParameters
from data.market_data import load_market_data
from pricing.black_scholes import black_scholes_fx_call
from volatility.heston import run_heston_simulation, heston_price, heston_smile

# --- Market Data & Parameters ---
S0, vol, _, date_from, date_to = load_market_data()
params = FXModelParameters(S0=S0, vol=vol, K=round(S0, 4), theta=vol**2, v0=vol**2)

# --- Heston Simulation & Pricing ---
paths, var_paths = run_heston_simulation(params, seed=SEED)
h_price  = heston_price(params, paths)
bs_price = black_scholes_fx_call(
    S0=params.S0, K=params.K,
    rd=params.domestic_r, rf=params.foreign_r,
    vol=params.vol, T=params.T
)

print(Fore.CYAN + "=" * 70 + Style.RESET_ALL)
print("HESTON MODEL")
print(f"{'Heston Price:':<20} {h_price:.6f}")
print(f"{'BS Price:':<20} {bs_price:.6f}")
print(Fore.GREEN + f"{'Difference:':<20} {h_price - bs_price:.6f}" + Style.RESET_ALL)
print(Fore.CYAN + "=" * 70 + Style.RESET_ALL)

# --- Volatility Smile ---
strikes   = np.linspace(params.K * 0.90, params.K * 1.10, 15)
smile_ivs = heston_smile(params, strikes, paths)

# --- BS vs Heston Prices over Strikes ---
bs_prices = np.array([
    black_scholes_fx_call(S0=params.S0, K=k, rd=params.domestic_r,
                          rf=params.foreign_r, vol=params.vol, T=params.T)
    for k in strikes])
heston_prices = np.array([
    heston_price(FXModelParameters(**{**params.__dict__, 'K': k}), paths)
    for k in strikes])

# --- Visualizations ---
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Heston Stochastic Volatility Model", fontsize=14)

ax1.plot(paths[:50, :].T, alpha=0.4, linewidth=0.8)
ax1.axhline(y=params.S0, color="black", linestyle="--", linewidth=1.5, label=f"S0 = {params.S0:.4f}")
ax1.axhline(y=params.K,  color="red",   linestyle="--", linewidth=1.5, label=f"Strike = {params.K:.4f}")
ax1.set_title("Heston Spot Paths")
ax1.set_xlabel("Time Steps (Days)")
ax1.set_ylabel("Spot Rate")
ax1.legend()

ax2.plot(np.sqrt(np.maximum(var_paths[:50, :], 0)).T, alpha=0.4, linewidth=0.8)
ax2.axhline(y=np.sqrt(params.theta), color="red", linestyle="--",
            linewidth=1.5, label=f"Long-run Vol = {np.sqrt(params.theta):.4f}")
ax2.set_title("Stochastic Volatility Paths")
ax2.set_xlabel("Time Steps (Days)")
ax2.set_ylabel("Volatility")
ax2.legend()

ax3.plot(strikes, smile_ivs, color="blue", marker="o", label="Heston Implied Vol")
ax3.axvline(x=params.K, color="red", linestyle="--", label=f"ATM = {params.K:.4f}")
ax3.set_title("Volatility Smile (Heston)")
ax3.set_xlabel("Strike")
ax3.set_ylabel("Implied Volatility")
ax3.legend()

ax4.plot(strikes, bs_prices,     color="orange", marker="o", label="BS Price")
ax4.plot(strikes, heston_prices, color="blue",   marker="o", label="Heston Price")
ax4.set_title("BS vs Heston Price over Strikes")
ax4.set_xlabel("Strike")
ax4.set_ylabel("Option Price")
ax4.legend()

plt.tight_layout()
plt.show()

# --- Sensitivity Analysis ---
rho_values  = [-0.7, -0.4, -0.1, 0.2, 0.5]
xi_values   = [0.1, 0.2, 0.3, 0.4, 0.5]
params_sens = FXModelParameters(**{**params.__dict__, 'MonteCarloSimulations': 10000})

fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Heston Sensitivity Analysis", fontsize=14)

for rho_val in rho_values:
    params_rho    = FXModelParameters(**{**params_sens.__dict__, 'rho': rho_val})
    paths_rho, _  = run_heston_simulation(params_rho, seed=SEED)
    ivs_rho       = heston_smile(params_rho, strikes, paths_rho)
    ax5.plot(strikes, ivs_rho, marker="o", markersize=3, label=f"ρ = {rho_val}")

ax5.axvline(x=params.K, color="black", linestyle="--", linewidth=0.8, label=f"ATM = {params.K:.4f}")
ax5.set_title(f"Volatility Smile for different ρ  |  ξ = {params.xi}")
ax5.set_xlabel("Strike")
ax5.set_ylabel("Implied Volatility")
ax5.legend()

for xi_val in xi_values:
    params_xi    = FXModelParameters(**{**params_sens.__dict__, 'xi': xi_val})
    paths_xi, _  = run_heston_simulation(params_xi, seed=SEED)
    ivs_xi       = heston_smile(params_xi, strikes, paths_xi)
    ax6.plot(strikes, ivs_xi, marker="o", markersize=3, label=f"ξ = {xi_val}")

ax6.axvline(x=params.K, color="black", linestyle="--", linewidth=0.8, label=f"ATM = {params.K:.4f}")
ax6.set_title(f"Volatility Smile for different ξ  |  ρ = {params.rho}")
ax6.set_xlabel("Strike")
ax6.set_ylabel("Implied Volatility")
ax6.legend()

plt.tight_layout()
plt.show()