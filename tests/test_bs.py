import numpy as np
from quant_engine.models.black_scholes import BlackScholesModel
from quant_engine.instruments.vanilla import EuropeanOption
from quant_engine.engines.analytical import AnalyticalEngine


def test_black_scholes_call_atm():
    """
    Test At-The-Money (ATM) Call option pricing against the known theoretical value.
    """
    model = BlackScholesModel(spot=100.0, rate=0.05, vol=0.20)
    call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
    engine = AnalyticalEngine()

    # The method returns a dictionary; extract the price
    result = engine.price_and_greeks(call, model)
    price = result['price']

    # An ATM Call (Spot 100, Strike 100, 1 year, 5% rate, 20% Vol) is worth ~10.4506
    assert np.isclose(price, 10.4506, atol=0.01), f"Expected ~10.45, got {price}"


def test_put_call_parity():
    """
    Verify that Put-Call Parity holds: C - P = S - K * exp(-rT)
    """
    model = BlackScholesModel(spot=100.0, rate=0.05, vol=0.20)
    call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
    put = EuropeanOption(strike=100.0, maturity=1.0, option_type='put')
    engine = AnalyticalEngine()

    # Extract prices from the returned dictionaries
    c_price = engine.price_and_greeks(call, model)['price']
    p_price = engine.price_and_greeks(put, model)['price']

    parity_left = c_price - p_price
    parity_right = model.spot - 100.0 * np.exp(-model.rate * 1.0)

    assert np.isclose(parity_left, parity_right, atol=0.001), "Put-Call parity failed"
