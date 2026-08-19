import os
import sys

# Add the project root to the Python path to allow absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from quant_engine.instruments.autocall import Autocall
from quant_engine.models.black_scholes import BlackScholesModel
from quant_engine.engines.monte_carlo import MonteCarloEngine


def main():
    print("--- Demo: Autocall Pricing (Memory vs. No Memory) ---")
    print("Simulating 5 years of daily market paths for 100,000 scenarios...\n")

    # 1. Market Model Setup
    # Standard equity market environment (Spot 100, 5% Risk-Free Rate, 20% Volatility)
    model = BlackScholesModel(spot=100.0, rate=0.05, vol=0.20)

    # 2. Product Specifications
    # 5-year product with annual observation dates
    maturity = 5.0
    obs_times = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Creation of two identical products, differing only by the memory effect.
    print("-Initializing Autocall structures...")

    autocall_memory = Autocall(
        notional=100.0,
        initial_spot=100.0,
        maturity=maturity,
        observation_times=obs_times,
        autocall_level=100.0,  # Market needs to be at 100% of strike to trigger early redemption
        coupon_level=80.0,  # Market needs to be above 80% to pay the coupon
        protection_level=60.0,  # European Down-and-In barrier protecting capital up to -40% drop
        coupon_rate=0.08,  # 8% annual coupon
        risk_free_rate=model.rate,
        memory_effect=True
    )

    autocall_no_memory = Autocall(
        notional=100.0,
        initial_spot=100.0,
        maturity=maturity,
        observation_times=obs_times,
        autocall_level=100.0,
        coupon_level=80.0,
        protection_level=60.0,
        coupon_rate=0.08,
        risk_free_rate=model.rate,
        memory_effect=False
    )

    # 3. Monte Carlo Engine Setup
    # 5 years = 5 * 252 trading days. Using 100,000 paths for high precision.
    engine = MonteCarloEngine(num_paths=100000, time_steps=252 * 5)

    # 4. Pricing Execution
    print("\n-Pricing Autocall with memory effect... please wait.")
    price_memory = engine.price(instrument=autocall_memory, model=model)

    print("-Pricing Autocall without memory effect... please wait.")
    price_no_memory = engine.price(instrument=autocall_no_memory, model=model)

    # 5. Financial Analysis & Results
    print("\n--- Fair value results ---")
    print(f"Price with memory    : {price_memory:.2f}%")
    print(f"Price without memory : {price_no_memory:.2f}%")

    memory_premium = price_memory - price_no_memory
    print(f"\n--- Financial insight ---")
    print(f"Memory premium : +{memory_premium:.2f}%")
    print("Conclusion: The memory feature allows investors to recover missed coupons.")
    print("Because it strictly increases the expected payout, it mathematically increases the fair value of the product.")


if __name__ == "__main__":
    main()
