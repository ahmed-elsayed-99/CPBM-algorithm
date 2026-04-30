from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cpbm",
    version="0.1.0",
    author="Ahmed Elsayed",
    author_email="",
    description="Capital-Pulse Behavioral Model for purchase-trend prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ahmed-elsayed-99/CPBM-algorithm",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "torch>=2.0.0",
        "torch-geometric>=2.4.0",
        "networkx>=3.1",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
        "tqdm>=4.65.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
)