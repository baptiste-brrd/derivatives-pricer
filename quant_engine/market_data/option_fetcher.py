import os
import yfinance as yf
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

class MarketDataFetcher:
    """
    Extrait les chaînes d'options depuis Yahoo Finance et applique
    les filtres de liquidité essentiels pour la modélisation.
    """

    def __init__(self, ticker_symbol: str):
        self.ticker_symbol = ticker_symbol
        self.ticker = yf.Ticker(ticker_symbol)

        # Récupération directe du prix spot
        self.spot_price = float(self.ticker.history(period='1d')['Close'].iloc[-1])

        # Chemin du fichier cache
        current_file_path = os.path.abspath(__file__)
        market_data_dir = os.path.dirname(current_file_path)
        quant_engine_dir = os.path.dirname(market_data_dir)
        project_root = os.path.dirname(quant_engine_dir)
        self.cache_dir = os.path.join(project_root, "data")
        self.cache_file = os.path.join(self.cache_dir, f"{self.ticker_symbol}_options_cache.csv")

        # Vérification que le dossier data/ existe
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def fetch_options(self, min_volume: int = 10, use_cache: bool = True) -> pd.DataFrame:
        """
        Récupère les options. Utilise le fichier local si use_cache=True et que le fichier existe.
        """

        # Lecture du cache
        if use_cache and os.path.exists(self.cache_file):
            print(f"Chargement des données depuis le cache local : {self.cache_file}")
            df = pd.read_csv(self.cache_file)
            return df

        # Si le cache est désactivé ou absent
        print("Téléchargement des données depuis Yahoo Finance...")
        expirations = self.ticker.options
        if not expirations:
            raise ValueError(f"Aucune option disponible pour {self.ticker_symbol}")

        options_data = []

        for exp_date in expirations:
            opt_chain = self.ticker.option_chain(exp_date)

            calls = opt_chain.calls.copy()
            calls['Option_Type'] = 'call'

            puts = opt_chain.puts.copy()
            puts['Option_Type'] = 'put'

            chain = pd.concat([calls, puts], ignore_index=True)

            # Calcul du time to maturity (T) en années
            exp_datetime = datetime.strptime(exp_date, "%Y-%m-%d")
            chain['T'] = (exp_datetime - datetime.now()).days / 365.25

            # Supprime l'option des données si elle est échue
            if chain['T'].iloc[0] <= 0:
                continue

            chain['Expiration'] = exp_date
            options_data.append(chain)

        df = pd.concat(options_data, ignore_index=True)

        # Calculs de la moneyness et du mid price
        df['Moneyness'] = df['strike'] / self.spot_price
        df['Mid_Price'] = (df['bid'] + df['ask']) / 2.0

        # Filtre de liquidité pour ne garder que les options les plus échangées
        mask_valid = (
                (df['volume'] >= min_volume) &
                (df['bid'] > 0) &
                (df['ask'] > 0)
        )
        df = df[mask_valid].copy()

        columns = ['Expiration', 'T', 'strike', 'Option_Type', 'Moneyness', 'bid', 'ask', 'Mid_Price',
                   'impliedVolatility']

        # Sauvegarde les données dans le fichier cache
        print(f"Sauvegarde des données dans le cache : {self.cache_file}")
        df_final = df[columns]
        df_final.to_csv(self.cache_file, index=False)

        return df_final
