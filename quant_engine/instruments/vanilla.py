import numpy as np
from quant_engine.instruments.base_instrument import BaseInstrument


class EuropeanOption(BaseInstrument):
    """
    Standard European Call or Put option.
    """

    def __init__(self, strike: float, maturity: float, option_type: str = 'call'):
        """
        Initializes the European option's characteristics.

        Args:
            strike (float): The strike price (K).
            maturity (float): Time to maturity in years (T).
            option_type (str): 'call' or 'put'.
        """
        self.strike = strike
        self.maturity = maturity
        self.option_type = option_type.lower()

        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be strictly 'call' or 'put'.")

    def get_payoff(self, spot_paths: np.ndarray) -> np.ndarray:
        """
        Calculates the European option payoff at maturity.
        Since it is a European option, it is not path-dependent.
        Only the final spot price (at the last time step) matters.
        """
        # Extract the final spot prices from all simulated paths
        # (taking the last column of the 2D array)
        final_spots = spot_paths[:, -1]

        # Vectorized payoff calculation for maximum performance
        if self.option_type == 'call':
            return np.maximum(final_spots - self.strike, 0.0)
        elif self.option_type == 'put':
            return np.maximum(self.strike - final_spots, 0.0)
