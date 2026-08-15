import numpy as np
from abc import ABC, abstractmethod


class BaseInstrument(ABC):
    """
    Abstract base class for all financial instruments.
    Any specific product (Vanilla, Barrier, Autocall) must inherit from this class
    and implement the get_payoff method.
    """

    @abstractmethod
    def get_payoff(self, spot_paths: np.ndarray) -> np.ndarray:
        """
        Calculates the payoff of the instrument given simulated spot paths.

        Args:
            spot_paths (np.ndarray): A 2D array of simulated price paths generated
                                     by the Monte Carlo engine.
                                     Shape: (number_of_paths, number_of_time_steps).

        Returns:
            np.ndarray: A 1D array containing the final payoff for each simulated path.
        """
        pass
