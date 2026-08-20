import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from quant_engine.market_data.option_fetcher import MarketDataFetcher
from quant_engine.market_data.vol_solver import VolatilitySolver
from quant_engine.instruments.vanilla import EuropeanOption
from quant_engine.models.black_scholes import BlackScholesModel
from quant_engine.engines.monte_carlo import MonteCarloEngine


def main():
    print("--- Demo: End-to-End Market Data to Pricing ---")
    ticker = "^SPX"

    # 1. Seamless Data Pipeline (Fetch -> Solve)
    print("Fetching and solving market data in memory...")
    fetcher = MarketDataFetcher(ticker)
    raw_data = fetcher.fetch_options(use_cache=True)

    solver = VolatilitySolver(rate=0.05)
    market_df = solver.solve_surface(df=raw_data, spot_price=fetcher.spot_price)

    # 2. Selecting a realistic At-The-Money (ATM) option for testing
    calls = market_df[market_df['Option_Type'] == 'call'].copy()
    calls['Moneyness_Diff'] = abs(calls['Moneyness'] - 1.0)
    test_opt = calls.sort_values('Moneyness_Diff').iloc[0]

    strike = test_opt['strike']
    maturity = test_opt['T']
    real_vol = test_opt['My_Implied_Vol']
    market_price = test_opt['Mid_Price']

    print(f"\nSelected Real Market Option ({ticker}):")
    print(f"    - Spot: {fetcher.spot_price:.2f} $")
    print(f"    - Strike: {strike} $")
    print(f"    - Maturity: {maturity:.2f} Years")
    print(f"    - Implied Volatility (Numba): {real_vol * 100:.2f}%")
    print(f"    - Quoted Market Price: {market_price:.2f} $")

    # 3. Object Integration & Pricing
    my_call = EuropeanOption(strike=strike, maturity=maturity, option_type='call')
    my_model = BlackScholesModel(spot=fetcher.spot_price, rate=0.05, vol=real_vol)

    # Using 100 time steps for a fast vanilla European pricing
    mc_engine = MonteCarloEngine(num_paths=100000, time_steps=100)

    print("\nEngine running Stochastic Monte Carlo paths...")
    mc_price = mc_engine.price(instrument=my_call, model=my_model)

    # 4. Results comparison
    print(f"\n>>> Model Calculated Price: {mc_price:.2f} $")
    print(f">>> Actual Market Price:    {market_price:.2f} $")
    print(f">>> Absolute Difference:    {abs(mc_price - market_price):.2f} $")


if __name__ == "__main__":
    main()
