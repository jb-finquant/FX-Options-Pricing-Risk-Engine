import numpy as np
from scipy.stats import norm
from model.parameters import FXModelParameters
from simulation.gbm import run_simulation


def compute_mc_var(params: FXModelParameters, mc_price: float, seed: int = 2003) -> tuple:
    """
    Computes Monte Carlo VaR for a European FX option over a one-day horizon.

    Simulates spot paths over the risk horizon, revalues the option via
    Garman-Kohlhagen, and derives the loss distribution.

    References:
    -----------
    Hull, J. (2022). Options, Futures, and Other Derivatives. Chapter 22.
    """
    params_horizon = FXModelParameters(**{**params.__dict__, 'T': params.time_horizon})
    paths          = run_simulation(params_horizon, seed=seed)
    S_T            = paths[:, -1]
    T_remaining    = params.T - params.time_horizon

    d1 = (np.log(S_T / params.K) + (params.domestic_r - params.foreign_r + 0.5 * params.vol ** 2) * T_remaining) / (params.vol * np.sqrt(T_remaining))
    d2 = d1 - params.vol * np.sqrt(T_remaining)

    if params.OptionsType == "Call":
        price_horizon = (S_T * np.exp(-params.foreign_r * T_remaining) * norm.cdf(d1)
                         - params.K * np.exp(-params.domestic_r * T_remaining) * norm.cdf(d2))
    elif params.OptionsType == "Put":
        price_horizon = (params.K * np.exp(-params.domestic_r * T_remaining) * norm.cdf(-d2)
                         - S_T * np.exp(-params.foreign_r * T_remaining) * norm.cdf(-d1))
    else:
        raise ValueError(f"Unknown option type: '{params.OptionsType}'. Expected 'Call' or 'Put'.")

    pnl = price_horizon - mc_price
    var = np.quantile(pnl, params.confidence_level)
    return var, pnl


def compute_parametric_var(pnl: np.ndarray, confidence_level: float) -> float:
    """
    Computes parametric VaR assuming normally distributed P&L.

    References:
    -----------
    Hull, J. (2022). Options, Futures, and Other Derivatives. Chapter 22.
    """
    mu    = np.mean(pnl)
    sigma = np.std(pnl)
    return mu + norm.ppf(confidence_level) * sigma


def compute_historical_var(pnl: np.ndarray, confidence_level: float) -> float:
    """
    Computes historical VaR via empirical quantile of the return distribution.

    References:
    -----------
    Hull, J. (2022). Options, Futures, and Other Derivatives. Chapter 22.
    """
    return np.quantile(pnl, confidence_level)