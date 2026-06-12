import numpy as np
from scipy.optimize import brentq
from model.parameters import FXModelParameters
from pricing.black_scholes import black_scholes_fx_call


def implied_vol(market_price: float, params: FXModelParameters) -> float:
    """
    Computes implied volatility via Brent's root-finding method.

    Finds σ such that BS(σ) = market price. Brent's method combines the
    reliability of bisection with the speed of inverse quadratic interpolation,
    making it more robust than Newton-Raphson (requires derivative, can diverge)
    and faster than bisection (linear vs superlinear convergence).

    Returns np.nan if no solution exists within the search interval [1e-5, 5.0].

    References:
    -----------
    Brent, R.P. (1973). Algorithms for Minimization without Derivatives.
    Prentice-Hall, New Jersey. Chapter 4.

    Press, W.H. et al. (2007). Numerical Recipes. Cambridge University Press. Chapter 9.
    """
    def objective(sigma: float) -> float:
        return black_scholes_fx_call(
            S0=params.S0, K=params.K,
            rd=params.domestic_r, rf=params.foreign_r,
            vol=sigma, T=params.T
        ) - market_price

    try:
        return brentq(objective, 1e-5, 5.0)
    except ValueError:
        return np.nan