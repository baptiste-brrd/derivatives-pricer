import numpy as np
from quant_engine.instruments.base_instrument import BaseInstrument

class BarrierOption(BaseInstrument):
    """
    Class representing a Barrier Option (Up-and-Out, Down-and-Out, Up-and-In, Down-and-In).
    The payoff depends not only on the final price but on the entire price path.
    """

    def __init__(self, strike: float, maturity: float, option_type: str,
                 barrier_level: float, barrier_type: str):
        """
        Initializes the Barrier Option.

        Args:
            strike (float): The strike price (K).
            maturity (float): Time to maturity in years (T).
            option_type (str): 'call' or 'put'.
            barrier_level (float): The price level that triggers the barrier event (B).
            barrier_type (str): The type of barrier.
                                Options: 'uo' (Up-and-Out), 'do' (Down-and-Out),
                                         'ui' (Up-and-In), 'di' (Down-and-In).
        """
        self.strike = strike
        self.maturity = maturity
        self.option_type = option_type.lower()
        self.barrier_level = barrier_level
        self.barrier_type = barrier_type.lower()

        # Validation
        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'.")
        if self.barrier_type not in ['uo', 'do', 'ui', 'di']:
            raise ValueError("barrier_type must be 'uo', 'do', 'ui', or 'di'.")

    def get_payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Calculates the payoff for a given set of simulated price paths.

        Args:
            paths (np.ndarray): 2D array of simulated paths. Shape: (num_paths, time_steps + 1)

        Returns:
            np.ndarray: 1D array of payoffs for each path.
        """
        # 1. Extract the final prices for the vanilla payoff calculation
        final_prices = paths[:, -1]

        # 2. Calculate the base Vanilla Payoff (as if there was no barrier)
        if self.option_type == 'call':
            vanilla_payoffs = np.maximum(final_prices - self.strike, 0.0)
        else:
            vanilla_payoffs = np.maximum(self.strike - final_prices, 0.0)

        # 3. Analyze the entire trajectory to check for barrier breaches
        # We find the maximum and minimum price reached on EACH path
        max_reached = np.max(paths, axis=1)
        min_reached = np.min(paths, axis=1)

        # 4. Apply the specific barrier logic using NumPy vectorization
        if self.barrier_type == 'uo': # Up-and-Out
            # Payoff is vanilla IF the max price never breached the barrier, else 0
            survived = max_reached < self.barrier_level
            payoffs = np.where(survived, vanilla_payoffs, 0.0)

        elif self.barrier_type == 'do': # Down-and-Out
            # Payoff is vanilla IF the min price never dropped below the barrier, else 0
            survived = min_reached > self.barrier_level
            payoffs = np.where(survived, vanilla_payoffs, 0.0)

        elif self.barrier_type == 'ui': # Up-and-In
            # Payoff is vanilla ONLY IF the max price breached the barrier, else 0
            activated = max_reached >= self.barrier_level
            payoffs = np.where(activated, vanilla_payoffs, 0.0)

        elif self.barrier_type == 'di': # Down-and-In
            # Payoff is vanilla ONLY IF the min price dropped below the barrier, else 0
            activated = min_reached <= self.barrier_level
            payoffs = np.where(activated, vanilla_payoffs, 0.0)

        return payoffs
