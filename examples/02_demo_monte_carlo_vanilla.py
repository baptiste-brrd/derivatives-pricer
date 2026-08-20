import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from quant_engine.instruments.vanilla import EuropeanOption
from quant_engine.models.black_scholes import BlackScholesModel
from quant_engine.engines.monte_carlo import MonteCarloEngine


def main():
    print("--- Demo: Object-Oriented Monte Carlo Pricing ---")

    # 1. Structuring the product (The Contract)
    # A standard European Call Option, Strike=100, 1 Year to maturity
    my_call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
    print(f"Instrument Created: {my_call.option_type.upper()} | Strike: {my_call.strike} | T: {my_call.maturity}Y")

    # 2. Defining the market dynamics (The Underlying Asset)
    # Spot=100, Risk-Free Rate=5%, Volatility=20%
    my_model = BlackScholesModel(spot=100.0, rate=0.05, vol=0.20)
    print(f"Market Model Created: Black-Scholes | Spot: {my_model.spot} | Vol: {my_model.vol * 100}%")

    # 3. Configuring the simulation engine
    # 100,000 parallel universes, observed daily (252 steps)
    num_paths = 100000
    mc_engine = MonteCarloEngine(num_paths=num_paths, time_steps=252)
    print(f"Engine Configured: Monte Carlo ({num_paths:,} paths)")

    # 4. Pricing Execution
    print("\nRunning simulations... please wait.")
    price = mc_engine.price(instrument=my_call, model=my_model)

    print(f"\n>>> Calculated Option Fair Value: ${price:.4f}")


if __name__ == "__main__":
    main()
