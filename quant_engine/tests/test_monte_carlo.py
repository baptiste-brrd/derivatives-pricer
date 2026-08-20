import numpy as np
from quant_engine.models.black_scholes import BlackScholesModel
from quant_engine.instruments.vanilla import EuropeanOption
from quant_engine.engines.analytical import AnalyticalEngine
from quant_engine.engines.monte_carlo import MonteCarloEngine


def test_monte_carlo_convergence():
    """
    Verify that the Monte Carlo engine converges to the analytical Black-Scholes price
    for a standard Vanilla option.
    """
    model = BlackScholesModel(spot=100.0, rate=0.05, vol=0.20)
    call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')

    # Exact analytical price (extracting from dict)
    analytical_engine = AnalyticalEngine()
    exact_price = analytical_engine.price_and_greeks(call, model)['price']

    # Monte Carlo simulated price
    mc_engine = MonteCarloEngine(num_paths=100000, time_steps=252)
    np.random.seed(42)
    mc_price = mc_engine.price(call, model)

    error_margin = exact_price * 0.01
    assert abs(
        mc_price - exact_price) < error_margin, f"Monte Carlo price ({mc_price}) diverges from theoretical value ({exact_price})"
