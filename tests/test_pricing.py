import pytest
from model.parameters import FXModelParameters
from simulation.gbm import run_simulation
from pricing.monte_carlo import compute_price
from pricing.black_scholes import black_scholes_fx_call


def make_params():
    """Standard EUR/USD-like test parameters."""
    return FXModelParameters(S0=1.10, K=1.10, vol=0.08)


def test_mc_price_converges_to_black_scholes():
    """MC pricer should converge to the closed-form GK/BS price for large N."""
    params = make_params()

    paths = run_simulation(params, seed=42)
    mc_price = compute_price(params, paths)

    bs_price = black_scholes_fx_call(
        S0=params.S0, K=params.K,
        rd=params.domestic_r, rf=params.foreign_r,
        vol=params.vol, T=params.T
    )

    assert mc_price == pytest.approx(bs_price, abs=0.01)


def test_bs_price_is_positive():
    """Sanity check: option price should never be negative."""
    params = make_params()
    price = black_scholes_fx_call(
        S0=params.S0, K=params.K,
        rd=params.domestic_r, rf=params.foreign_r,
        vol=params.vol, T=params.T
    )
    assert price > 0


def test_bs_call_at_the_money_below_spot():
    """ATM call price should be less than the spot rate (basic no-arbitrage sanity check)."""
    params = make_params()
    price = black_scholes_fx_call(
        S0=params.S0, K=params.K,
        rd=params.domestic_r, rf=params.foreign_r,
        vol=params.vol, T=params.T
    )
    assert price < params.S0


def test_put_call_parity():
    """C - P = S0*e^(-rf*T) - K*e^(-rd*T) — fundamental no-arbitrage relationship."""
    ...