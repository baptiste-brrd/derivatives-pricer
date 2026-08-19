import numpy as np
from quant_engine.instruments.base_instrument import BaseInstrument


class MonteCarloEngine:
    """
    Pricing engine using Monte Carlo simulation.
    """

    def __init__(self, num_paths: int = 100000, time_steps: int = 252):
        """
        Initializes the Monte Carlo engine.

        Args:
            num_paths (int): Number of scenarios (paths) to simulate.
            time_steps (int): Number of time steps (e.g., 252 for daily steps over 1 year).
        """
        self.num_paths = num_paths
        self.time_steps = time_steps

    def price(self, instrument: BaseInstrument, model) -> float:
        """
        Calculates the fair price of a financial instrument.

        Args:
            instrument (BaseInstrument): The financial product to price (e.g., Call, Put, Autocall).
            model: The market dynamics model used to generate paths (e.g., BlackScholesModel).

        Returns:
            float: The estimated price (Present Value) of the instrument.
        """
        # Generate future price paths based on the instrument's maturity
        paths = model.generate_paths(
            maturity=instrument.maturity,
            num_paths=self.num_paths,
            time_steps=self.time_steps
        )

        # Calculate the payoff for all these simulated paths
        payoffs = instrument.get_payoff(paths)

        # Calculate the average (expected value) of all these future payoffs
        expected_payoff = np.mean(payoffs)

        # Discount the expected payoff back to its present value using the risk-free rate
        if getattr(instrument, 'is_pre_discounted', False):
            # Cash flows occur at various dates and have already been discounted internally by the instrument
            present_value = expected_payoff
        else:
            discount_factor = np.exp(-model.rate * instrument.maturity)
            present_value = expected_payoff * discount_factor

        return float(present_value)
