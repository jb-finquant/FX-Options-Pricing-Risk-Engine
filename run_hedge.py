import matplotlib.pyplot as plt
from colorama import Fore, Style

from config import SEED
from model.parameters import FXModelParameters
from data.market_data import load_market_data
from simulation.gbm import run_simulation
from hedging.delta_hedge import run_delta_hedge

# --- Market Data & Parameters ---
S0, vol, _, date_from, date_to = load_market_data()
params = FXModelParameters(S0=S0, vol=vol, K=round(S0, 4))

# --- Simulation ---
paths = run_simulation(params, seed=SEED)
path  = paths[0]

# --- Delta Hedging ---
position = -1  # -1 = Short Call, +1 = Long Call
result   = run_delta_hedge(params, path, position=position)

position_label = "Short Call" if position == -1 else "Long Call"

print(Fore.CYAN + "=" * 70 + Style.RESET_ALL)
print(f"DELTA HEDGING SIMULATION — {position_label}")
print(f"{'Final Hedging Error:':<27} {result['hedging_error'][-1]:.6f}")
print(f"{'Final Cumulative Hedge P&L:':<27} {result['hedge_pnl'][-1]:.6f}")
print(Fore.CYAN + "=" * 70 + Style.RESET_ALL)

# --- Visualizations ---
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle(f"Delta Hedging Simulation — {position_label}", fontsize=14)

ax1.plot(result["spot"], color="blue")
ax1.axhline(y=params.K, color="red", linestyle="--", label=f"Strike = {params.K:.4f}")
ax1.set_title("Spot Path")
ax1.set_xlabel("Time Steps (Days)")
ax1.set_ylabel("Spot Rate")
ax1.legend()

ax2.plot(result["delta"], color="orange")
ax2.set_title("Delta over Time")
ax2.set_xlabel("Time Steps (Days)")
ax2.set_ylabel("Delta")

ax3.plot(result["option_pnl"],    color="green", label="Cumulative Option P&L")
ax3.plot(result["hedge_pnl"],     color="blue",  label="Cumulative Hedge P&L")
ax3.plot(result["hedging_error"], color="red",   label="Hedging Error (Net)")
ax3.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
ax3.set_title("P&L Attribution")
ax3.set_xlabel("Time Steps (Days)")
ax3.set_ylabel("P&L")
ax3.legend()

ax4.plot(result["hedging_error"], color="red",    label="Hedging Error")
ax4.plot(result["theta_pnl"],     color="blue",   label="Cumulative Theta P&L")
ax4.plot(result["gamma_pnl"],     color="orange", label="Cumulative Gamma P&L")
ax4.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
ax4.set_title("P&L Attribution: Theta vs Gamma")
ax4.set_xlabel("Time Steps (Days)")
ax4.set_ylabel("P&L")
ax4.legend()

plt.tight_layout()
plt.savefig("docs/images/delta_hedging.png", dpi=150, bbox_inches="tight")
plt.show()