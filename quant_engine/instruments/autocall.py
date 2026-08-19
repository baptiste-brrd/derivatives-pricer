import numpy as np
from quant_engine.instruments.base_instrument import BaseInstrument


class Autocall(BaseInstrument):
    """
    Class representing an Autocallable structured product.
    Includes early redemption, conditional coupons, memory effect, and capital protection.
    """

    def __init__(self, notional: float, initial_spot: float, maturity: float,
                 observation_times: list, autocall_level: float, coupon_level: float,
                 protection_level: float, coupon_rate: float,
                 risk_free_rate: float, memory_effect: bool = True):
        """
        Initializes the Autocall product.

        Args:
            notional (float): The invested capital (e.g., 1000 or 100).
            initial_spot (float): The reference spot price at inception (Strike).
            maturity (float): Total time to maturity in years.
            observation_times (list): List of observation dates in years (e.g., [1.0, 2.0, 3.0]).
            autocall_level (float): Price triggering early redemption.
            coupon_level (float): Price triggering coupon payment.
            protection_level (float): European barrier for capital protection at maturity.
            coupon_rate (float): The coupon value as a percentage of notional (e.g., 0.08 for 8%).
            risk_free_rate (float): Rate used to discount cash flows occurring at different times.
            memory_effect (bool): If True, unpaid coupons are stored and paid later when conditions are met.
        """
        self.notional = notional
        self.initial_spot = initial_spot
        self.maturity = maturity
        self.observation_times = sorted(observation_times)

        self.autocall_level = autocall_level
        self.coupon_level = coupon_level
        self.protection_level = protection_level
        self.coupon_amount = self.notional * coupon_rate
        self.risk_free_rate = risk_free_rate
        self.memory_effect = memory_effect

        self.is_pre_discounted = True

    def get_payoff(self, paths: np.ndarray, dt: float = None) -> np.ndarray:
        """
        Calculates the discounted payoff for each simulated path.

        Args:
            paths (np.ndarray): 2D array of simulated paths. Shape: (num_paths, time_steps + 1)
            dt (float): Time step duration in years (e.g., 1/252).

        Returns:
            np.ndarray: 1D array of PRESENT VALUE payoffs for each path.
        """
        num_paths, total_steps = paths.shape

        # If dt is not provided, we infer it from the maturity and number of steps
        if dt is None:
            dt = self.maturity / (total_steps - 1)

        # Arrays to track the state of each path
        discounted_payoffs = np.zeros(num_paths)
        active_paths = np.ones(num_paths, dtype=bool)  # True = Not yet autocalled
        accumulated_coupons = np.zeros(num_paths)  # To track missed coupons (Memory effect)

        for obs_time in self.observation_times:
            # Find the exact array index corresponding to this observation date
            idx = int(round(obs_time / dt))
            idx = min(idx, total_steps - 1)  # Safety to avoid out of bounds

            current_prices = paths[:, idx]
            is_maturity = (obs_time == self.observation_times[-1])

            # 1. Check Coupon Conditions (for active paths only)
            coupon_hit = (current_prices >= self.coupon_level) & active_paths

            if self.memory_effect:
                # Add this year's coupon to the memory pot
                accumulated_coupons = np.where(active_paths, accumulated_coupons + self.coupon_amount,
                                               accumulated_coupons)
                # Calculate what we pay today (everything in the pot if hit, 0 otherwise)
                payout_coupon = np.where(coupon_hit, accumulated_coupons, 0.0)
                # Empty the pot for paths that just got paid
                accumulated_coupons = np.where(coupon_hit, 0.0, accumulated_coupons)
            else:
                payout_coupon = np.where(coupon_hit, self.coupon_amount, 0.0)

            # 2. Check Autocall Conditions
            if not is_maturity:
                autocall_hit = (current_prices >= self.autocall_level) & active_paths

                # If Autocalled: We pay Notional + Coupon, discount it, and kill the path
                cash_flow = np.where(autocall_hit, self.notional + payout_coupon, 0.0)
                discounted_payoffs += cash_flow * np.exp(-self.risk_free_rate * obs_time)

                # If NOT Autocalled but Coupon Hit: We pay just the coupon and continue
                cash_flow_coupon_only = np.where(coupon_hit & ~autocall_hit, payout_coupon, 0.0)
                discounted_payoffs += cash_flow_coupon_only * np.exp(-self.risk_free_rate * obs_time)

                # Deactivate autocalled paths
                active_paths &= ~autocall_hit

            # 3. Maturity Logic (The Final Day)
            else:
                # Capital Protection check (European Down-and-In Put logic)
                capital_saved = (current_prices >= self.protection_level) & active_paths
                capital_lost = (current_prices < self.protection_level) & active_paths

                # Good scenario: Capital is intact
                final_cash_flow = np.where(capital_saved, self.notional + payout_coupon, 0.0)

                # Bad scenario: Tracker effect (loss of capital), no coupon
                tracker_payoff = self.notional * (current_prices / self.initial_spot)
                final_cash_flow += np.where(capital_lost, tracker_payoff, 0.0)

                # Discount the final cash flows and add to the total
                discounted_payoffs += final_cash_flow * np.exp(-self.risk_free_rate * obs_time)

                # Deactivate all remaining paths
                active_paths[:] = False

        return discounted_payoffs
