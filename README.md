# Derivatives Pricer
Python library for exotic derivatives pricing and quantitative risk management.

## Overview
This project is a modular Python-based library designed to price complex financial derivatives (with a focus on Autocallables) and extract stable risk sensitivities (greeks). The architecture clearly separates financial instruments, market models, and computational engines, allowing for scalable pricing logic and efficient risk analysis.

## Key Features

### 1. Market Data & Volatility Surface (`quant_engine/market_data/`)
*   **Data Fetching:** Tools to retrieve live options market data.
*   **Surface Calibration:** Construction and interpolation of implied volatility surfaces from market quotes, allowing the engine to move beyond flat volatility assumptions.

### 2. Financial Instruments (`quant_engine/instruments/`)
*   **Vanilla Options:** Standard European Calls & Puts.
*   **Exotics (Autocalls):** Complex structured products featuring:
    *   Early redemption conditions (Autocall barriers).
    *   Conditional coupons with **Memory Effect**.
    *   Capital protection via European Down-and-In Put barriers.
    *   *Design Pattern:* Internal cash-flow discounting capabilities communicated to engines via Duck Typing (`is_pre_discounted`), preventing double-discounting anomalies.

### 3. Computational Engines (`quant_engine/engines/`)
*   **Analytical Engine:** Closed-form Black-Scholes-Merton solutions for standard derivatives.
*   **Monte Carlo Engine:** Highly optimized, vectorized stochastic simulator using `NumPy` for path-dependent payoff evaluation over multi-year horizons (e.g., 100,000 paths over 5 years computed in seconds).

### 4. Risk Management (`quant_engine/risk/`)
*   **Numerical Greeks (Bump & Reprice):** Extraction of delta, vega, and rho using finite central difference schemes.
*   **Common Random Numbers (CRN):** Implementation of fixed random seeds across spot/volatility shocks to eliminate stochastic noise and guarantee numerically stable greeks.
*   **Vega Profile Analysis:** Tools to visualize the dynamic nature of an Autocall's Vega (Convexity & Pin Risk), demonstrating the shift from *Short Vega* at inception to *Long Vega* near observation thresholds.

## Project Architecture

```text
volatility-surface-pricer/
│
├── data/                       # Local market data storage
│
├── examples/                   # Executable scripts for demonstration
│   ├── 01_volatility_surface.py
│   ├── 02_monte_carlo_vanilla.py
│   ├── 03_end_to_end_pricing.py
│   ├── 04_analytical_vs_mc.py
│   ├── 05_barrier_vs_vanilla.py
│   ├── 06_autocall_memory.py
│   └── 07_autocall_greeks.py
│
├── quant_engine/               # Core library
│   ├── market_data/
│   │   ├── option_fetcher.py
│   │   ├── vol_solver.py
│   │   └── vol_surface.py
│   ├── instruments/
│   │   ├── base_instrument.py
│   │   ├── vanilla.py
│   │   ├── barrier.py
│   │   └── autocall.py
│   ├── models/
│   │   └── black_scholes.py
│   ├── engines/
│   │   ├── analytical.py
│   │   └── monte_carlo.py
│   └── risk/
│       └── risk_management.py
│
├── tests/                      # Unit testing suite
│   ├── test_bs.py
│   ├── test_monte_carlo.py
│   └── test_vol_surface.py
│
├── README.md
├── requirements.txt            # Project dependencies
└── setup.py                    # Package configuration
```
## Technical Highlights for Code Reviewers
*   **Memory Efficiency:** Monte Carlo paths are evaluated using 100% vectorized NumPy arrays, avoiding slow Python `for` loops on the scenario axis.
*   **Immutable Market States:** The Risk Manager utilizes `copy.deepcopy` to simulate market shocks (Up/Down bumps) in strict isolation, preserving the integrity of the base market model.
*   **Quantitative Pragmatism:** Second-order greeks (gamma) and time-derivatives (theta) were deliberately excluded from the Monte Carlo numerical extraction to avoid the mathematical instability of double-differentiation on simulated noise, favoring analytical approximations where necessary.

## Tech Stack
*   **Language:** Python 3.10+
*   **Core Libraries:** `NumPy` (vectorized computation), `Pandas` (data manipulation), `SciPy` (optimization & statistics)
*   **Performance:** `Numba` (JIT compilation utilized to accelerate implied volatility root-finding algorithms)
*   **Testing:** pytest
