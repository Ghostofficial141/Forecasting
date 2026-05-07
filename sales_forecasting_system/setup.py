"""
setup.py
========
Package setup for the Sales Forecasting System.
Allows `pip install -e .` for editable development installs.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="sales-forecasting-system",
    version="1.0.0",
    author="ML Engineering Team",
    author_email="ml-team@company.com",
    description=(
        "Production-ready time-series sales forecasting system "
        "with SARIMA, Prophet, XGBoost, and LSTM"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "forecast-train=main:run_training",
            "forecast-predict=main:run_prediction",
            "forecast-api=main:run_api",
        ],
    },
)
