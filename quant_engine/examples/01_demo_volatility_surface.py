import os
import sys

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from quant_engine.market_data.option_fetcher import MarketDataFetcher
from quant_engine.market_data.vol_solver import VolatilitySolver
from quant_engine.market_data.vol_surface import VolatilitySurface


def main():
    print("--- Demo: 3D Implied Volatility Surface ---")
    ticker = "^SPX"

    # 1. Fetching market data (using cache for speed in demos)
    print(f"Fetching options chain for {ticker}...")
    fetcher = MarketDataFetcher(ticker)
    raw_data = fetcher.fetch_options(use_cache=True)
    spot = fetcher.spot_price

    # 2. Solving implied volatility using the C-compiled Numba engine
    print("Running Newton-Raphson solver...")
    solver = VolatilitySolver(rate=0.05)
    enriched_data = solver.solve_surface(df=raw_data, spot_price=spot)

    # 3. Generating the 3D plot
    print("Generating 3D surface plot. Close the window to end the script.")
    surface = VolatilitySurface(enriched_data)
    surface.plot_surface(resolution=50)


if __name__ == "__main__":
    main()
