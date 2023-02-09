from setuptools import setup, find_packages

INSTALL_DEPENDENCIES = [
    "pip>=22.1.0",
    "numpy>=1.22.0",
    "pandas>=1.4.0",
    "pmdarima>=2.0.0",
    "pyspark>=3.2.1",
    "scikit-learn>=1.0.0",
    "scipy>=1.7.0",
    "prophet>=1.0",
    "pyyaml>=5.0",
    "xgboost>=1.5.0",
    "lightgbm>=3.3.0"
]

TEST_DEPENDENCIES = [
    "pytest",
    "pytest-cov",
    "pytest-mock",
    "pyspark-test",
]

EXTRA_DEPENDENCIES = [
    "matplotlib",
    "jupyter",
    "ipykernel",
    "ipython"
]

setup(
    name="tsff",
    version="0.1.0",
    python_requires=">=3.8",
    packages=find_packages(include=['tsff', 'tsff.*']),
    install_requires=INSTALL_DEPENDENCIES,
    extras_require={'interactive': EXTRA_DEPENDENCIES, 'test':TEST_DEPENDENCIES}
)
