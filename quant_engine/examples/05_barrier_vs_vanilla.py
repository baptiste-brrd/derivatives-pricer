import os
import sys

# Add the project root to the Python path to allow absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from quant_engine.instruments.vanilla import EuropeanOption
from quant_engine.instruments.barrier import BarrierOption
from quant_engine.models.black_scholes import BlackScholesModel
from quant_engine.engines.monte_carlo import MonteCarloEngine


def main():
    print("Pricing Test: Vanilla Option vs. Barrier Option")

    # 1. Market Model (Spot 100, Vol 20%, Risk-Free Rate 5%)
    my_model = BlackScholesModel(spot=100.0, rate=0.05, vol=0.20)

    # 2. Instruments (Same Strike, Same Maturity)
    # Standard European Call
    vanilla_call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')

    # Barrier Call (Up-and-Out deactivating at 120)
    barrier_call = BarrierOption(
        strike=100.0,
        maturity=1.0,
        option_type='call',
        barrier_level=120.0,
        barrier_type='uo'
    )

    # 3. Pricing Engine (100,000 paths, daily observation/252 steps)
    engine = MonteCarloEngine(num_paths=100000, time_steps=252)

    # 4. Pricing Execution
    print("\n-Pricing the Vanilla Call... please wait.")
    price_vanilla = engine.price(instrument=vanilla_call, model=my_model)

    print(f"-Pricing the Barrier Call (Up-and-Out {barrier_call.barrier_level:.1f})... please wait.")
    price_barrier = engine.price(instrument=barrier_call, model=my_model)

    # 5. Results Output
    print("\nRESULTS:")
    print(f"Vanilla Call Price : {price_vanilla:.4f} $")
    print(f"Barrier Call Price : {price_barrier:.4f} $")

    # Calculate the price discount due to the added risk of the barrier
    discount = ((price_vanilla - price_barrier) / price_vanilla) * 100
    print(f"Price Reduction    : {discount:.2f} %")


if __name__ == "__main__":
    main()
