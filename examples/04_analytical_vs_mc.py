import os
import sys

# Add the project root to the Python path to allow absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import core objects
from quant_engine.instruments.vanilla import EuropeanOption
from quant_engine.models.black_scholes import BlackScholesModel
from quant_engine.engines.monte_carlo import MonteCarloEngine
from quant_engine.engines.analytical import AnalyticalEngine


def main():
    print("--- Demo: Analytical vs. Monte Carlo Engine Comparison ---")

    # The instrument - Call Option, Strike 100, 1 Year Maturity
    my_call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')

    # The model (Market Dynamics) - Spot 100, Risk-Free Rate 5%, Volatility 20%
    # THIS EXACT OBJECT IS SHARED ACROSS BOTH ENGINES
    my_model = BlackScholesModel(spot=100.0, rate=0.05, vol=0.20)

    # 1. Analytical Pricing (The Mathematical Truth)
    print("\n[1/2] Running Analytical Engine (Exact Black-Scholes)...")
    analytical_engine = AnalyticalEngine()

    # Pass the instrument and model
    analytical_results = analytical_engine.price_and_greeks(instrument=my_call, model=my_model)

    exact_price = analytical_results['price']
    print(f"      -> Exact Calculated Price : {exact_price:.4f} $")

    # 2. Monte Carlo Pricing (The Stochastic Approximation)
    print("\n[2/2] Running Monte Carlo Engine (100,000 paths)...")
    mc_engine = MonteCarloEngine(num_paths=100000, time_steps=252)

    # Pass EXACTLY THE SAME ARGUMENTS to the Monte Carlo engine
    mc_price = mc_engine.price(instrument=my_call, model=my_model)

    print(f"      -> Estimated Price : {mc_price:.4f} $")

    # 3. Comparison
    diff = abs(exact_price - mc_price)
    print("\n--- COMPARISON ---")
    print(f"Absolute difference between theory and simulation : {diff:.4f} $")

    if diff < 0.05:
        print("SUCCESS: The Monte Carlo engine is perfectly calibrated!")
    else:
        print("WARNING: The difference is slightly high, consider increasing the number of paths.")


if __name__ == "__main__":
    main()
