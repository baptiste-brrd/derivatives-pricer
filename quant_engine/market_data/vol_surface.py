import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


class VolatilitySurface:
    """
    Constructs and visualizes a 3D implied volatility surface
    from market data via interpolation.
    """

    def __init__(self, options_df: pd.DataFrame):
        # Keep the entire DataFrame because the fetcher filtered for OTM Puts and OTM Calls
        self.data = options_df.copy()

        # Remove null or anomalous values
        self.data = self.data.dropna(subset=['My_Implied_Vol'])
        self.data = self.data[self.data['My_Implied_Vol'] > 0.01]
        self.data = self.data[self.data['My_Implied_Vol'] < 2.00]

        # Filter out ultra-short maturities (e.g., < 7 days) to remove market noise
        self.data = self.data[self.data['T'] > 0.02]

        # Calculate Log-Moneyness ln(K/S) to ensure mathematical symmetry
        self.data['Log_Moneyness'] = np.log(self.data['Moneyness'])

    def plot_surface(self, resolution: int = 50):
        """
        Generates the 3D plot of the volatility surface.

        Args:
            resolution (int): The number of points on the X and Y axes for smoothing.
                              A higher number results in a finer mesh.
        """

        # Extracting the 3 dimensions (X, Y, Z)
        # Using log-moneyness on the X-axis for structural consistency with Black-Scholes
        X_points = self.data['Log_Moneyness'].values
        Y_points = self.data['T'].values
        Z_points = self.data['My_Implied_Vol'].values

        # Creating a blank canvas (Meshgrid)
        # Defining the boundaries of the plot based on market data extremes
        X_min, X_max = np.min(X_points), np.max(X_points)
        Y_min, Y_max = np.min(Y_points), np.max(Y_points)

        # np.linspace creates 'resolution' evenly spaced points between min and max
        X_grid, Y_grid = np.meshgrid(
            np.linspace(X_min, X_max, resolution),
            np.linspace(Y_min, Y_max, resolution)
        )

        # Interpolation
        # griddata takes the scattered points and computes the height (Z) for each point on the grid
        # The 'cubic' method smooths the surface dynamically
        Z_grid = griddata(
            points=(X_points, Y_points),
            values=Z_points,
            xi=(X_grid, Y_grid),
            method='linear'
        )

        # 3D Plotting with Matplotlib
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # plot_surface draws the mesh. The 'cmap' adds colors (e.g., blue for low, yellow for high)
        surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', edgecolor='none', alpha=0.9)

        # Overlaying the actual market points to visually verify the interpolation fit
        ax.scatter(X_points, Y_points, Z_points, color='red', s=10, label='Market Data')

        ax.set_title('Implied Volatility Surface (SPY)')
        ax.set_xlabel('Log-Moneyness ln(K/S)')
        ax.set_ylabel('Time to Maturity (Years)')
        ax.set_zlabel('Implied Volatility')

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Vol Level')
        plt.legend()
        plt.show()
