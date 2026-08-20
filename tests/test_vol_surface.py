import numpy as np
from quant_engine.market_data.vol_solver import implied_vol_newton_jit


def test_implied_volatility_solver():
    """
    Test if the implied volatility solver correctly reverse-engineers the volatility
    from a given option price using vectorized inputs.
    """
    spot = 100.0  # float (scalar)
    strike = 100.0
    rate = 0.05   # float (scalar)
    maturity = 1.0
    known_target_price = 10.4506

    # Strictly respect the function's type signatures:
    # arrays for vectorized inputs, standard floats for Spot and Rate
    calculated_vols = implied_vol_newton_jit(
        np.array([known_target_price]),  # target_prices (array)
        spot,                            # S (float)
        np.array([strike]),              # K (array)
        np.array([maturity]),            # T (array)
        rate,                            # r (float)
        np.array([1])                    # is_call_array (array)
    )

    result = calculated_vols[0]

    assert result > 0, "Implied volatility must be strictly positive"
    assert abs(result - 0.20) < 0.005, f"Solver failed to find correct vol. Expected ~0.20, got {result}"
