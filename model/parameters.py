from dataclasses import dataclass, field


@dataclass
class FXModelParameters:
    """
    Parameters for FX options pricing and risk engine.

    Market inputs (required):
        S0:  Current spot rate
        K:   Strike price
        vol: Annualized historical volatility (from market data)

    Rate parameters:
        domestic_r: USD risk-free rate (Fed Funds Rate)
        foreign_r:  EUR risk-free rate (ECB Rate)

    Heston stochastic volatility parameters:
        kappa: Mean reversion speed         — Gatheral (2006)
        theta: Long-run variance            — set from realized variance
        xi:    Volatility of volatility     — Gatheral (2006)
        rho:   Spot-vol correlation         — typical FX estimate
        v0:    Initial variance             — set from realized variance
    """
    # --- Required ---
    S0:  float
    K:   float
    vol: float

    # --- Rate Parameters ---
    domestic_r: float = 0.0525
    foreign_r:  float = 0.04

    # --- Option Parameters ---
    T:                      float = 1.0
    OptionsType:            str   = "Call"
    MonteCarloSimulations:  int   = 100_000
    time_steps:             int   = 252
    confidence_level:       float = 0.05
    time_horizon:           float = 1 / 252

    # --- Heston Parameters ---
    kappa: float = 2.0
    theta: float = None
    xi:    float = 0.3
    rho:   float = -0.3
    v0:    float = None