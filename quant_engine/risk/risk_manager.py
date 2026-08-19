import numpy as np
import copy


class RiskManager:
    """
    Risk Management module using Finite Differences (Bump & Reprice).
    Extracts greeks (delta, vega, rho) for complex instruments using a Monte Carlo engine.
    """

    def __init__(self, engine, seed: int = 42):
        """
        Args:
            engine: The pricing engine (e.g., MonteCarloEngine).
            seed (int): The random seed to ensure Common Random Numbers (CRN) across bumps.
        """
        self.engine = engine
        self.seed = seed

    def compute_greeks(self, instrument, model, bump_size: float = 0.01) -> dict:
        """
        Computes delta, vega, and rho using a central difference scheme.

        Args:
            instrument: The financial product (e.g., Autocall).
            model: The market model (e.g., BlackScholesModel).
            bump_size (float): The relative bump size for the spot (1% by default).

        Returns:
            dict: The instrument's base price, Delta, Vega, and Rho.
        """
        # 1. Base Price
        np.random.seed(self.seed)
        base_price = self.engine.price(instrument, model)

        # DELTA CALCULATION (Spot Bump)
        original_spot = model.spot
        dS = original_spot * bump_size

        model_up_spot = copy.deepcopy(model)
        model_up_spot.spot = original_spot + dS
        np.random.seed(self.seed)  # CRN
        price_up_spot = self.engine.price(instrument, model_up_spot)

        model_down_spot = copy.deepcopy(model)
        model_down_spot.spot = original_spot - dS
        np.random.seed(self.seed)  # CRN
        price_down_spot = self.engine.price(instrument, model_down_spot)

        delta = (price_up_spot - price_down_spot) / (2 * dS)

        # VEGA CALCULATION (Volatility Bump)
        original_vol = model.vol
        dVol = 0.01  # Absolute bump of 1 volatility point

        model_up_vol = copy.deepcopy(model)
        model_up_vol.vol = original_vol + dVol
        np.random.seed(self.seed)  # CRN
        price_up_vol = self.engine.price(instrument, model_up_vol)

        model_down_vol = copy.deepcopy(model)
        model_down_vol.vol = original_vol - dVol
        np.random.seed(self.seed)  # CRN
        price_down_vol = self.engine.price(instrument, model_down_vol)

        vega = (price_up_vol - price_down_vol) / (2 * dVol)

        # RHO CALCULATION (Interest Rate Bump)
        original_rate = model.rate
        dRate = 0.01  # Absolute bump of 100 basis points (1%)

        model_up_rate = copy.deepcopy(model)
        model_up_rate.rate = original_rate + dRate
        np.random.seed(self.seed)  # CRN
        price_up_rate = self.engine.price(instrument, model_up_rate)

        model_down_rate = copy.deepcopy(model)
        model_down_rate.rate = original_rate - dRate
        np.random.seed(self.seed)  # CRN
        price_down_rate = self.engine.price(instrument, model_down_rate)

        rho = (price_up_rate - price_down_rate) / (2 * dRate)

        # Return standardized greeks (vega and rho per 1% move)
        return {
            'price': float(base_price),
            'delta': float(delta),
            'vega': float(vega / 100),
            'rho': float(rho / 100)
        }
