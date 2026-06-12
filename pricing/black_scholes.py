import numpy as np
from scipy.stats import norm


def black_scholes_fx_call(S0: float, K: float, rd: float, rf: float, vol: float, T: float) -> float:
    """
    Prices a European FX call option via the Garman-Kohlhagen (1983) model.

    Extends Black-Scholes to FX by treating the foreign interest rate
    as a continuous dividend yield on the underlying currency.

    References:
    -----------
    Garman, M.B. & Kohlhagen, S.W. (1983). Foreign Currency Option Values.
    Journal of International Money and Finance, 2(3), 231–237.
    """
    d1 = (np.log(S0 / K) + (rd - rf + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S0 * np.exp(-rf * T) * norm.cdf(d1) - K * np.exp(-rd * T) * norm.cdf(d2)