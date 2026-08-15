import numpy as np

class BlackScholesModel:
    """
    Trajectory simulator using the Black-Scholes model (constant volatility).
    Stochastic Differential Equation (SDE): dS = r*S*dt + sigma*S*dW
    """

    def __init__(self, spot: float, rate: float, vol: float):
        """
        Initializes the Black-Scholes model parameters.

        Args:
            spot (float): The current spot price of the underlying asset (S0).
            rate (float): The risk-free interest rate (r).
            vol (float): The implied volatility (sigma).
        """
        self.spot = spot
        self.rate = rate
        self.vol = vol

    def generate_paths(self, maturity: float, num_paths: int, time_steps: int) -> np.ndarray:
        """
        Generates a matrix of simulated price paths.

        Args:
            maturity (float): Time to maturity in years (T).
            num_paths (int): Number of scenarios to simulate.
            time_steps (int): Number of time steps per path.

        Returns:
            np.ndarray: A 2D array representing the simulated price paths.
                        Shape: (num_paths, time_steps + 1)
        """
        dt = maturity / time_steps

        # 1. Generate random shocks (Standard Normal Distribution)
        # Draw num_paths * time_steps random numbers
        Z = np.random.standard_normal((num_paths, time_steps))

        # 2. Prepare the price matrix (initialized with zeros)
        paths = np.zeros((num_paths, time_steps + 1))

        # 3. Set the starting point (T=0) to the current Spot price for all paths
        paths[:, 0] = self.spot

        # 4. Vectorized mathematical engine (Geometric Brownian Motion)
        # Calculate the growth factor for each time step
        growth_factor = np.exp((self.rate - 0.5 * self.vol ** 2) * dt + self.vol * np.sqrt(dt) * Z)

        # Use np.cumprod to accumulate the compounding growth across the entire timeline
        paths[:, 1:] = self.spot * np.cumprod(growth_factor, axis=1)

        return paths
