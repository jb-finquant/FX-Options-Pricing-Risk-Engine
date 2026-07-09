import numpy as np
from model.parameters import FXModelParameters


def payoff(params: FXModelParameters, S_t: np.ndarray) -> np.ndarray:
    """
    Computes option payoff at expiry.

    References:
    -----------
    Hull, J. (2022). Options, Futures, and Other Derivatives. Chapter 9.
    """
    if params.OptionsType == "Call":
        return np.maximum(S_t - params.K, 0.0)
    elif params.OptionsType == "Put":
        return np.maximum(params.K - S_t, 0.0)
    else:
        raise ValueError(f"Unknown option type: '{params.OptionsType}'. Expected 'Call' or 'Put'.")


def compute_price(params: FXModelParameters, paths: np.ndarray) -> float:
    """
    Prices a European FX option via Monte Carlo simulation.

    Discounts the expected payoff under the risk-neutral measure
    using the domestic risk-free rate.

    References:
    -----------
    Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. Chapter 1.
    """
    S_t     = paths[:, -1]
    payoffs = payoff(params, S_t)
    return np.mean(payoffs) * np.exp(-params.domestic_r * params.T)