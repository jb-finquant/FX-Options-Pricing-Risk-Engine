import numpy as np
from model.parameters import FXModelParameters


def run_simulation(params: FXModelParameters, seed: int = 2003) -> np.ndarray:
    """
    Simulates FX spot price paths via Geometric Brownian Motion with Antithetic Variates.

    The log-price increment is decomposed into a deterministic drift component
    and a stochastic diffusion component.

    Uses antithetic variates for variance reduction by pairing each random
    draw Z with its mirror -Z, halving the effective standard error.

    References:
    -----------
    Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. Chapter 4.
    """
    rng          = np.random.default_rng(seed)
    dt           = params.T / params.time_steps
    mu           = params.domestic_r - params.foreign_r
    fx_drift     = (mu - 0.5 * params.vol ** 2) * dt
    fx_diffusion = params.vol * np.sqrt(dt)

    Z      = rng.standard_normal((params.MonteCarloSimulations // 2, params.time_steps))
    Z_full = np.concatenate([Z, -Z], axis=0)

    increments = fx_drift + fx_diffusion * Z_full
    log_S      = np.log(params.S0) + np.cumsum(increments, axis=1)

    return np.exp(log_S)