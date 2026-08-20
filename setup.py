from setuptools import setup, find_packages

setup(
    name="derivatives-pricer",
    version="1.0.0",
    author="Baptiste Briard",
    description="Derivatives pricing and risk management engine.",
    packages=find_packages(include=["quant_engine", "quant_engine.*"]),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "yfinance",
        "numba"
    ],
    python_requires=">=3.9",
)
