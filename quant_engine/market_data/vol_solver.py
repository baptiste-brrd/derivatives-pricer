import numpy as np
import pandas as pd
import math
from numba import njit

# NUMBA CORE (Low-level C-compiled functions)

@njit(fastmath=True)
def bs_price_and_vega_jit(S: float, K: float, T: float, r: float, sigma: float, is_call: bool):
    """
    Ultra-fast unit engine to calculate the Black-Scholes price and Vega.
    Uses 'math' instead of 'np' because Numba optimizes pure scalars better.
    """
    # Mathematical safety: avoid division by zero
    if T <= 0 or sigma <= 0:
        if is_call:
            return max(S - K, 0.0), 0.0
        else:
            return max(K - S, 0.0), 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Ultra-fast approximation of the Normal distribution CDF via math.erf
    cdf_d1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    cdf_d2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))

    if is_call:
        price = S * cdf_d1 - K * math.exp(-r * T) * cdf_d2
    else:
        price = K * math.exp(-r * T) * (1.0 - cdf_d2) - S * (1.0 - cdf_d1)

    # Analytical calculation of Vega: d(Price)/d(Vol)
    pdf_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2.0 * math.pi)
    vega = S * math.sqrt(T) * pdf_d1

    return price, vega


@njit(fastmath=True)
def implied_vol_newton_jit(target_prices: np.ndarray, S: float, K: np.ndarray, T: np.ndarray,
                           r: float, is_call_array: np.ndarray, tol: float = 1e-5, max_iter: int = 100):
    """
    Vectorized Newton-Raphson solver. Applies the algorithm across thousands of options.
    """
    n = len(target_prices)
    implied_vols = np.full(n, 0.20)  # Initial guess: 20% volatility for all options

    for i in range(n):
        if T[i] <= 0:
            implied_vols[i] = np.nan
            continue

        sigma = implied_vols[i]

        # Newton-Raphson Loop
        for _ in range(max_iter):
            price, vega = bs_price_and_vega_jit(S, K[i], T[i], r, sigma, is_call_array[i])
            diff = price - target_prices[i]

            # Stop if the calculated price is close enough to the market price
            if abs(diff) < tol:
                break

            # Prevent division by zero if Vega is too flat
            if vega < 1e-8:
                break

            # Newton's formula: X_new = X_old - f(X)/f'(X)
            sigma = sigma - (diff / vega)

            # Volatility cannot be negative
            if sigma <= 0.0001:
                sigma = 0.0001

        implied_vols[i] = sigma

    return implied_vols

# Wrapper to use as a function

class VolatilitySolver:
    """
    Interface to process market data DataFrames and compute
    implied volatilities using the low-level Numba JIT solver.
    """

    def __init__(self, rate: float = 0.05):
        """
        Args:
            rate (float): The risk-free interest rate (r) used for pricing.
        """
        self.rate = rate

    def solve_surface(self, df: pd.DataFrame, spot_price: float) -> pd.DataFrame:
        """
        Takes a DataFrame of options, extracts the required arrays, runs the Numba
        solver, and appends the calculated implied volatilities.

        Args:
            df (pd.DataFrame): Market data containing at least 'Mid_Price', 'strike', 'T', 'Option_Type'.
            spot_price (float): The current price of the underlying asset.

        Returns:
            pd.DataFrame: A copy of the input DataFrame with a new 'My_Implied_Vol' column.
        """
        # Create a copy to avoid altering the original fetched data
        result_df = df.copy()

        # Numba requires strict boolean arrays, not strings like 'call' or 'put'
        is_call_array = (result_df['Option_Type'].str.lower() == 'call').values

        # Extract underlying NumPy arrays for maximum speed
        target_prices = result_df['Mid_Price'].values
        K = result_df['strike'].values
        T = result_df['T'].values

        print(f"Solving implied volatility for {len(result_df)} options using Newton-Raphson...")

        # Call the C-compiled Numba engine
        calculated_vols = implied_vol_newton_jit(
            target_prices=target_prices,
            S=spot_price,
            K=K,
            T=T,
            r=self.rate,
            is_call_array=is_call_array
        )

        # Inject the results back into the DataFrame
        result_df['My_Implied_Vol'] = calculated_vols

        return result_df
