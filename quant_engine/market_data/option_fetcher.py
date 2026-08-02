import os
import yfinance as yf
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

class MarketDataFetcher:
    """
    Extracts option chains from Yahoo Finance and applies
    essential liquidity filters for modeling.
    """

    def __init__(self, ticker_symbol: str):
        self.ticker_symbol = ticker_symbol
        self.ticker = yf.Ticker(ticker_symbol)

        # Direct retrieval of the spot price
        self.spot_price = float(self.ticker.history(period='1d')['Close'].iloc[-1])

        # Cache file path configuration
        current_file_path = os.path.abspath(__file__)
        market_data_dir = os.path.dirname(current_file_path)
        quant_engine_dir = os.path.dirname(market_data_dir)
        project_root = os.path.dirname(quant_engine_dir)
        self.cache_dir = os.path.join(project_root, "data")
        self.cache_file = os.path.join(self.cache_dir, f"{self.ticker_symbol}_options_cache.csv")

        # Ensure the data/ directory exists, create it otherwise
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def fetch_options(self, min_volume: int = 10, use_cache: bool = True) -> pd.DataFrame:
        """
        Retrieves options. Uses the local file if use_cache=True and the file exists.
        """

        # Read from cache
        if use_cache and os.path.exists(self.cache_file):
            print(f"Loading data from local cache: {self.cache_file}")
            df = pd.read_csv(self.cache_file)
            return df

        # If cache is disabled or missing
        print("Downloading data from Yahoo Finance...")
        expirations = self.ticker.options
        if not expirations:
            raise ValueError(f"No options available for {self.ticker_symbol}")

        options_data = []

        for exp_date in expirations:
            opt_chain = self.ticker.option_chain(exp_date)

            calls = opt_chain.calls.copy()
            calls['Option_Type'] = 'call'

            puts = opt_chain.puts.copy()
            puts['Option_Type'] = 'put'

            chain = pd.concat([calls, puts], ignore_index=True)

            # Compute time to maturity (T) in years
            exp_datetime = datetime.strptime(exp_date, "%Y-%m-%d")
            chain['T'] = (exp_datetime - datetime.now()).days / 365.25

            # Drop expired options
            if chain['T'].iloc[0] <= 0:
                continue

            chain['Expiration'] = exp_date
            options_data.append(chain)

        df = pd.concat(options_data, ignore_index=True)

        # Compute moneyness and mid price
        df['Moneyness'] = df['strike'] / self.spot_price
        df['Mid_Price'] = (df['bid'] + df['ask']) / 2.0

        # Liquidity filter to keep most exchanged options
        mask_valid = (
                (df['volume'] >= min_volume) &
                (df['bid'] > 0) &
                (df['ask'] > 0)
        )
        df = df[mask_valid].copy()

        columns = ['Expiration', 'T', 'strike', 'Option_Type', 'Moneyness', 'bid', 'ask', 'Mid_Price',
                   'impliedVolatility']

        # Save data into the cache file
        print(f"Saving data to cache: {self.cache_file}")
        df_final = df[columns]
        df_final.to_csv(self.cache_file, index=False)

        return df_final
