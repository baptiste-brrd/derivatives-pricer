import numpy as np
from scipy.stats import norm
from quant_engine.instruments.vanilla import EuropeanOption


class AnalyticalEngine:
    """
    Pricing engine based on the exact closed-form Black-Scholes formula.
    Calculates both the theoretical fair value and the greeks.
    """

    def price_and_greeks(self, instrument: EuropeanOption, model) -> dict:
        """
        Calculates the exact price and the main Greeks (Delta, Gamma, Vega, Theta, Rho).

        Args:
            instrument (EuropeanOption): The contract to price (Call or Put).
            model: The market dynamics model (must contain spot, rate, vol).

        Returns:
            dict: A dictionary containing the price and the greeks.
        """
        S = model.spot
        K = instrument.strike
        T = instrument.maturity
        r = model.rate
        sigma = model.vol

        # Safety: Handle expired options (Intrinsic value only)
        if T <= 0.0:
            payoff = max(S - K, 0.0) if instrument.option_type == 'call' else max(K - S, 0.0)
            return {
                'price': payoff,
                'delta': 1.0 if (instrument.option_type == 'call' and S > K) else (-1.0 if S < K else 0.0),
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0,
                'rho': 0.0
            }

        # 1. Calculate d1 and d2 components
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # 2. Normal distribution Cumulative Density Functions (CDF) and Probability Density (PDF)
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        N_minus_d1 = norm.cdf(-d1)
        N_minus_d2 = norm.cdf(-d2)
        pdf_d1 = norm.pdf(d1)

        # 3. Calculate price and directional greeks based on option type
        if instrument.option_type == 'call':
            price = S * N_d1 - K * np.exp(-r * T) * N_d2
            delta = N_d1

            # Theta for a call option
            theta = (- (S * sigma * pdf_d1) / (2 * np.sqrt(T))
                     - r * K * np.exp(-r * T) * N_d2)

            # Rho for a call option
            rho = K * T * np.exp(-r * T) * N_d2

        elif instrument.option_type == 'put':
            price = K * np.exp(-r * T) * N_minus_d2 - S * N_minus_d1
            delta = N_d1 - 1.0

            # Theta for a put option
            theta = (- (S * sigma * pdf_d1) / (2 * np.sqrt(T))
                     + r * K * np.exp(-r * T) * N_minus_d2)

            # Rho for a put option
            rho = -K * T * np.exp(-r * T) * N_minus_d2

        else:
            raise ValueError("Unsupported option type. Must be 'call' or 'put'.")

        # 4. Gamma and Vega are identical for both calls and puts
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * pdf_d1

        # Return a complete dictionary with the greeks
        return {
            'price': float(price),
            'delta': float(delta),
            'gamma': float(gamma),
            'vega': float(vega / 100),  # Scaled to represent a 1% change in implied volatility
            'theta': float(theta / 365),  # Scaled to represent 1 calendar day of time decay
            'rho': float(rho / 100)  # Scaled to represent a 1% change in risk-free rate
        }
