import os
import sys

# Add the project root to the Python path to allow absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from quant_engine.instruments.autocall import Autocall
from quant_engine.models.black_scholes import BlackScholesModel
from quant_engine.engines.monte_carlo import MonteCarloEngine
from quant_engine.risk.risk_manager import RiskManager


def main():
    print("--- Demo: Autocall Risk Management (Greeks via Monte Carlo) ---")
    print("Computing delta, vega, and rho using Common Random Numbers (CRN)...\n")

    # 1. Core Setup
    # Market Environment
    model = BlackScholesModel(spot=100.0, rate=0.05, vol=0.20)

    # 5-Year Autocall with Memory Effect
    autocall = Autocall(
        notional=100.0,
        initial_spot=100.0,
        maturity=5.0,
        observation_times=[1.0, 2.0, 3.0, 4.0, 5.0],
        autocall_level=100.0,
        coupon_level=80.0,
        protection_level=60.0,
        coupon_rate=0.08,
        risk_free_rate=model.rate,
        memory_effect=True
    )

    # 2. Engine & Risk Manager Initialization
    # 100,000 paths for stability. Time steps = 5 years * 252 days
    engine = MonteCarloEngine(num_paths=100000, time_steps=252 * 5)

    # Initialize Risk Manager with a fixed seed to enforce CRN
    risk_manager = RiskManager(engine=engine, seed=42)

    # 3. Compute Greeks
    print("-Shocking market parameters and repricing (this involves 7 full Monte Carlo runs)...")
    greeks = risk_manager.compute_greeks(instrument=autocall, model=model)

    # 4. Risk Report Output
    print("\n--- Risk sensitivities report ---")
    print(f"Fair Value : {greeks['price']:.2f} %")
    print(f"Delta      : {greeks['delta']:.4f} (Change in price for 1€ increase in Spot)")
    print(f"Vega       : {greeks['vega']:.4f} (Change in price for 1% increase in Volatility)")
    print(f"Rho        : {greeks['rho']:.4f} (Change in price for 1% increase in Interest Rates)")

    print("\n--- Analysis ---")
    if greeks['vega'] < 0:
        print("Note: As expected, the Autocall is inherently short vega.")
        print("Higher volatility increases the probability of hitting the downside barrier, depreciating the product.")


if __name__ == "__main__":
    main()
