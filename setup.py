"""Package configuration for SmartRetail Analytics."""

from pathlib import Path
from setuptools import setup, find_packages

# Read the README for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path, "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="smartretail_analytics",
    version="0.1.0",
    author="SmartRetail Team",
    author_email="a.emad2152@nu.edu.eg",
    description=(
        "A comprehensive retail analytics solution for "
        "sales forecasting and inventory optimization"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ahmed1Osman/smartretail-analytics",
    packages=find_packages(include=["smartretail_analytics", "smartretail_analytics.*"]),
    package_data={
        "smartretail_analytics": ["data/*.csv", "models/*.joblib", "*.json"]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "mypy>=0.910",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-mock>=3.10.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "ipywidgets>=7.0.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "smartretail-train=smartretail_analytics.cli:train_cli",
            "smartretail-predict=smartretail_analytics.cli:predict_cli",
        ],
    },
)
