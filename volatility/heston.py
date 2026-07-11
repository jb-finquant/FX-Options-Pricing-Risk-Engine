import numpy as np
import warnings
from model.parameters import FXModelParameters
from volatility.implied_vol import implied_vol


def run_heston_simulation(params: FXModelParameters, seed: int = 2003) -> tuple:
    """
    Simulates spot and variance paths under the Heston (1993) stochastic
    volatility model via Euler-Maruyama discretisation.

    References:
    -----------
        Heston, S.L. (1993). A Closed-Form Solution for Options with
        Stochastic Volatility. Review of Financial Studies, 6(2), 327–343.
    """
    rng = np.random.default_rng(seed)

    N  = params.MonteCarloSimulations
    n  = params.time_steps
    dt = params.T / n
    mu = params.domestic_r - params.foreign_r
    v0    = params.v0    if params.v0    is not None else params.vol ** 2
    theta = params.theta if params.theta is not None else params.vol ** 2

    if 2 * params.kappa * theta <= params.xi ** 2:
        warnings.warn("Feller condition violated – variance may hit zero.")

    W1 = rng.standard_normal((N, n))
    W2 = rng.standard_normal((N, n))
    Z_v = W1
    Z_S = params.rho * W1 + np.sqrt(1 - params.rho ** 2) * W2

    paths     = np.zeros((N, n))
    var_paths = np.zeros((N, n))
    paths[:, 0]     = params.S0
    var_paths[:, 0] = v0

    for t in range(1, n):
        v = np.maximum(var_paths[:, t - 1], 0)

        var_paths[:, t] = (v
            + params.kappa * (theta - v) * dt
            + params.xi * np.sqrt(v * dt) * Z_v[:, t])

        paths[:, t] = paths[:, t - 1] * np.exp(
            (mu - 0.5 * v) * dt
            + np.sqrt(v * dt) * Z_S[:, t])

    return paths, var_paths


def heston_price(params: FXModelParameters, paths: np.ndarray) -> float:
    """
    Prices a European call option under Heston via Monte Carlo.
    """
    payoffs = np.maximum(paths[:, -1] - params.K, 0)
    return np.exp(-params.domestic_r * params.T) * np.mean(payoffs)


def heston_smile(params: FXModelParameters, strikes: np.ndarray,
                 paths: np.ndarray) -> np.ndarray:
    """
    Computes implied volatility smile from Heston MC prices.

    Prices the option at each strike using the same Heston paths,
    then extracts BS implied vol via Brent's method.
    """
    ivs = []
    for K in strikes:
        params_k = FXModelParameters(**{**params.__dict__, 'K': K})
        price    = heston_price(params_k, paths)
        iv       = implied_vol(price, params_k)
        ivs.append(iv)
    return np.array(ivs)