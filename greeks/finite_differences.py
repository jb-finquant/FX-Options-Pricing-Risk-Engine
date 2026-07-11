import numpy as np
from model.parameters import FXModelParameters
from scipy.stats import norm
from simulation.gbm import run_simulation
from pricing.monte_carlo import compute_price


def mc_delta(params: FXModelParameters, seed: int = 2003, h: float = None):
    """Delta via central finite differences: [V(S+h) - V(S-h)] / 2h - Glasserman (2003), Ch. 7."""
    if h is None:
        h = 0.01 * params.S0

    params_up = FXModelParameters(**{**params.__dict__, 'S0': params.S0 + h})
    params_dn = FXModelParameters(**{**params.__dict__, 'S0': params.S0 - h})

    price_up = compute_price(params_up, run_simulation(params_up, seed=seed))
    price_dn = compute_price(params_dn, run_simulation(params_dn, seed=seed))

    return (price_up - price_dn) / (2 * h)

def bs_delta(params: FXModelParameters):
    """Analytical Delta — Garman & Kohlhagen (1983)."""
    d1 = (np.log(params.S0 / params.K) + (params.domestic_r - params.foreign_r + 0.5 * params.vol**2) * params.T) / (params.vol * np.sqrt(params.T))

    if params.OptionsType == "Call":
        return np.exp(-params.foreign_r * params.T) * norm.cdf(d1)
    elif params.OptionsType == "Put":
        return -np.exp(-params.foreign_r * params.T) * norm.cdf(-d1)

def mc_gamma(params: FXModelParameters, seed: int = 2003, h: float = None):
    """Gamma via central finite differences: [V(S+h) - 2V(S) + V(S-h)] / h² — Glasserman (2003), Ch. 7."""
    if h is None:
        h = 0.01 * params.S0

    params_up   = FXModelParameters(**{**params.__dict__, 'S0': params.S0 + h})
    params_dn   = FXModelParameters(**{**params.__dict__, 'S0': params.S0 - h})

    price_up   = compute_price(params_up,   run_simulation(params_up,   seed=seed))
    price_base = compute_price(params,      run_simulation(params,      seed=seed))
    price_dn   = compute_price(params_dn,   run_simulation(params_dn,   seed=seed))

    return (price_up - 2 * price_base + price_dn) / (h ** 2)


def bs_gamma(params: FXModelParameters):
    """Gamma = e^(-rf·T)·N'(d1) / (S0·σ·√T) — Garman & Kohlhagen (1983)."""
    d1 = (np.log(params.S0 / params.K) + (params.domestic_r - params.foreign_r + 0.5 * params.vol**2) * params.T) / (params.vol * np.sqrt(params.T))
    return np.exp(-params.foreign_r * params.T) * norm.pdf(d1) / (params.S0 * params.vol * np.sqrt(params.T))


def mc_vega(params: FXModelParameters, seed: int = 2003, h: float = None):
    """Vega via central finite differences: [V(σ+h) - V(σ-h)] / 2h — Glasserman (2003), Ch. 7."""
    if h is None:
        h = 0.01 * params.vol

    params_up = FXModelParameters(**{**params.__dict__, 'vol': params.vol + h})
    params_dn = FXModelParameters(**{**params.__dict__, 'vol': params.vol - h})

    price_up = compute_price(params_up, run_simulation(params_up, seed=seed))
    price_dn = compute_price(params_dn, run_simulation(params_dn, seed=seed))

    return (price_up - price_dn) / (2 * h)


def bs_vega(params: FXModelParameters):
    """Vega = S0·e^(-rf·T)·N'(d1)·√T — Garman & Kohlhagen (1983)."""
    d1 = (np.log(params.S0 / params.K) + (params.domestic_r - params.foreign_r + 0.5 * params.vol**2) * params.T) / (params.vol * np.sqrt(params.T))
    return params.S0 * np.exp(-params.foreign_r * params.T) * norm.pdf(d1) * np.sqrt(params.T)


def mc_theta(params: FXModelParameters, seed: int = 2003, h: float = None):
    """Theta via central finite differences: [V(T-h) - V(T+h)] / 2h — Glasserman (2003), Ch. 7."""
    if h is None:
        h = 1 / 252

    params_up = FXModelParameters(**{**params.__dict__, 'T': params.T - h})
    params_dn = FXModelParameters(**{**params.__dict__, 'T': params.T + h})

    price_up = compute_price(params_up, run_simulation(params_up, seed=seed))
    price_dn = compute_price(params_dn, run_simulation(params_dn, seed=seed))

    return (price_up - price_dn) / (2 * h)


def bs_theta(params: FXModelParameters):
    """Analytical Theta — Garman & Kohlhagen (1983)."""
    d1 = (np.log(params.S0 / params.K) + (params.domestic_r - params.foreign_r + 0.5 * params.vol**2) * params.T) / (params.vol * np.sqrt(params.T))
    d2 = d1 - params.vol * np.sqrt(params.T)
    if params.OptionsType == "Call":
        return (
            -params.S0 * np.exp(-params.foreign_r * params.T) * norm.pdf(d1) * params.vol / (2 * np.sqrt(params.T))
            - params.domestic_r * params.K * np.exp(-params.domestic_r * params.T) * norm.cdf(d2)
            + params.foreign_r * params.S0 * np.exp(-params.foreign_r * params.T) * norm.cdf(d1)
        )
    elif params.OptionsType == "Put":
        return (
            -params.S0 * np.exp(-params.foreign_r * params.T) * norm.pdf(d1) * params.vol / (2 * np.sqrt(params.T))
            + params.domestic_r * params.K * np.exp(-params.domestic_r * params.T) * norm.cdf(-d2)
            - params.foreign_r * params.S0 * np.exp(-params.foreign_r * params.T) * norm.cdf(-d1)
        )


def mc_rho(params: FXModelParameters, seed: int = 2003, h: float = None):
    """Rho via central finite differences: [V(rd+h) - V(rd-h)] / 2h — Glasserman (2003), Ch. 7."""
    if h is None:
        h = 0.0001

    params_up = FXModelParameters(**{**params.__dict__, 'domestic_r': params.domestic_r + h})
    params_dn = FXModelParameters(**{**params.__dict__, 'domestic_r': params.domestic_r - h})

    price_up = compute_price(params_up, run_simulation(params_up, seed=seed))
    price_dn = compute_price(params_dn, run_simulation(params_dn, seed=seed))

    return (price_up - price_dn) / (2 * h)


def bs_rho(params: FXModelParameters):
    """Rho — Garman & Kohlhagen (1983)."""
    d1 = (np.log(params.S0 / params.K) + (params.domestic_r - params.foreign_r + 0.5 * params.vol**2) * params.T) / (params.vol * np.sqrt(params.T))
    d2 = d1 - params.vol * np.sqrt(params.T)
    if params.OptionsType == "Call":
        return params.K * params.T * np.exp(-params.domestic_r * params.T) * norm.cdf(d2)
    elif params.OptionsType == "Put":
        return -params.K * params.T * np.exp(-params.domestic_r * params.T) * norm.cdf(-d2)


def greek_over_spot(greek_fn, params: FXModelParameters, spot_range: np.ndarray) -> np.ndarray:
    """Evaluates a BS Greek function over a range of spot prices."""
    return np.array([greek_fn(FXModelParameters(**{**params.__dict__, 'S0': s})) for s in spot_range])