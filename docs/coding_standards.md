# Time Series Forecasting Accelerator (TSFA) coding standards

To improve readability and understanding, below are the coding standards expected for code contributions to TSFA:

1. The TSFA codebase will adhere to [PEP-8: Style Guide for Python Code](https://peps.python.org/pep-0008/). This is to ensure code readability. PEP-8 coding practices include coding standards such as:
    - Incorporating docstrings and typedefs for all classes, class methods, and functions.
    - Ensuring that lines are a maximum of 120 characters long.
    - Adhering to 4-space indentations.
    - Use human-understandable variable naming conventions.
    - Function and method names will be written in snake-case style (eg: `my_method_name()`), while class names will be written in Pascal-case style (eg: `MyClassName()`).
    - Constant names will be written in upper-case with underscores (eg: `MY_CONSTANT_NAME`).
    - Module names will be short lowercase words separated with underscores (eg: `one_hot_encoder.py`).
    - Package names will be short lowercase words with no underscores.
2. It is encouraged that literature references to sources that explain feature engineering approach, and ML algorithm is provided. Any assumptions made should also be detailed in the Docstring of the class or class method.
3. No "hardcoded" values. Use parameters that can be set in a configuration file.
4. Python package will comprise of Python scripts for utility Classes and functions. These scripts should met the Linting requirements. Unit tests (using Pytest) should be provided for core functionality.
5. When possible, consider using Pyspark for parallelization.
