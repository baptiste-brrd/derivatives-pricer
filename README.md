# Derivatives Pricer
Python library for derivatives pricing and delta-hedging simulation.

> **Status: Active Development / Work in Progress**
> *This repository contains the architecture and core modules of a derivatives pricing and risk management library. It is being developed as a preparatory project for advanced quantitative finance studies.*

## Overview
This project is a modular Python-based library designed to price complex financial derivatives and simulate dynamic hedging strategies. The primary focus is on understanding the mathematical mechanics behind volatility surfaces, stochastic volatility models (Heston), exotic options (Autocalls), and the practical realities of managing a trading book (Gamma Bleed).

## Key Features

### 1. Pricing Models (`quant_engine/models/` & `structuring/`)
*   **Vanilla Options:** Analytical pricing using the Black-Scholes-Merton framework.
*   **Stochastic Volatility:** Implementation of the Heston model for advanced pricing dynamics.
*   **Exotics (Autocalls):** Monte Carlo simulation engines to price structured products (including barrier conditions and memory effects).

### 2. Market Data & Volatility Surface (`quant_engine/market_data/`)
*   **Data Fetching:** Tools to retrieve live options market data.
*   **Surface Calibration:** Construction and interpolation of implied volatility surfaces from market data.

### 3. Hedging & Risk Management (`quant_engine/risk/`)
*   **Greeks Calculation:** Computation of first and second-order sensitivities (Delta, Gamma, Vega).
*   **Delta-Hedging Simulator:** A backtesting module to simulate discrete Delta-Hedging over time.
*   **Gamma Bleed Analysis:** Tracking the P&L impact of rebalancing costs and market friction against theoretical Gamma.

## Project Architecture

```text
volatility-surface-pricer/
│
├── data/                       # Local market data storage
│
├── examples/                   # Executable scripts for demonstration
│   ├── demo_autocall.py
│   ├── demo_fetcher.py
│   ├── demo_heston.py
│   └── hedging_demo.py
│
├── quant_engine/               # Core library
│   ├── market_data/
│   │   ├── option_fetcher.py
│   │   └── vol_surface.py
│   ├── models/
│   │   ├── base.py
│   │   ├── black_scholes.py
│   │   └── heston.py
│   ├── risk/
│   │   └── delta_hedging.py
│   └── structuring/
│       └── __init__.py
│
├── tests/                      # Unit testing suite
│   ├── test_bs.py
│   ├── test_monte_carlo.py
│   └── test_vol_surface.py
│
├── README.md
├── requirements.txt            # Project dependencies
├── run_tests.py                # Test execution script
└── setup.py                    # Package configuration
```

## Tech Stack
*   **Language:** Python 3.10+
*   **Core Libraries:** `NumPy`, `Pandas` (Data manipulation), `SciPy` (Optimization & Statistics)
*   **Performance:** `Numba` (JIT compilation for Monte Carlo optimization)
*   **Testing:** pytest
