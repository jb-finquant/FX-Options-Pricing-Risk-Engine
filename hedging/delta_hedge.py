import numpy as np
from model.parameters import FXModelParameters
from pricing.black_scholes import black_scholes_fx_call
from greeks.finite_differences import bs_delta, bs_gamma, bs_theta


def run_delta_hedge(params: FXModelParameters, path: np.ndarray, position: int = -1) -> dict:
    """
    Simulates a discrete delta hedging strategy for a short or long FX call option.

    At each time step, the hedge position is rebalanced to match the current
    BS delta. The P&L is decomposed into option P&L, hedge P&L, theta, and gamma
    components. The hedging error captures residual P&L from discrete rebalancing.

    References:
    -----------
    Hull, J. (2022). Options, Futures, and Other Derivatives. Chapter 19.
    """
    n_steps = len(path)
    dt      = params.T / n_steps

    spot             = np.zeros(n_steps)
    delta            = np.zeros(n_steps)
    option_val       = np.zeros(n_steps)
    hedge_pnl_daily  = np.zeros(n_steps)
    option_pnl_daily = np.zeros(n_steps)
    theta_pnl        = np.zeros(n_steps)
    gamma_pnl        = np.zeros(n_steps)
    total_pnl        = np.zeros(n_steps)

    for t in range(n_steps):
        S   = path[t]
        tau = max(params.T - t * dt, 1e-6)

        params_t = FXModelParameters(
            S0=S,
            K=params.K,
            vol=params.vol,
            domestic_r=params.domestic_r,
            foreign_r=params.foreign_r,
            T=tau
        )

        spot[t]       = S
        delta[t]      = bs_delta(params_t)
        option_val[t] = black_scholes_fx_call(
            S0=S, K=params.K,
            rd=params.domestic_r, rf=params.foreign_r,
            vol=params.vol, T=tau
        )

        if t > 0:
            dS                   = path[t] - path[t - 1]
            hedge_pnl_daily[t]   = -position * delta[t - 1] * dS
            d_option             = option_val[t] - option_val[t - 1]
            option_pnl_daily[t]  = position * d_option
            theta_pnl[t]         = position * bs_theta(params_t) * dt
            gamma_pnl[t]         = position * 0.5 * bs_gamma(params_t) * dS ** 2
            total_pnl[t]         = hedge_pnl_daily[t] + option_pnl_daily[t]

    return {
        "spot":          spot,
        "delta":         delta,
        "option_value":  option_val,
        "option_pnl":    np.cumsum(option_pnl_daily),
        "hedge_pnl":     np.cumsum(hedge_pnl_daily),
        "theta_pnl":     np.cumsum(theta_pnl),
        "gamma_pnl":     np.cumsum(gamma_pnl),
        "total_pnl":     total_pnl,
        "hedging_error": np.cumsum(total_pnl),
    }