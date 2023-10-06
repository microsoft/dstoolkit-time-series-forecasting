# Time Series Forecasting Accelerator (TSFA)

## Overview

### Objective

- Provide a set of utilities and tools for time series forecasting.
- Capable of training models at scale using Pyspark.
- Build these standardized tools into a shared package for reuse.

### Source code extensibility

- Contributing Data Scientists can add new functionality such as feature engineering and machine learning algorithms.
- The codebase can evolve with new functionality as Data Scientists experiment with new feature engineering approaches and machine learning algorithms.

### The accelerator comprises of

- JSON configuration file: Used to provide parameters for dataset, feature engineering, model training and evaluation. Sample configuration files are provided in the `notebooks/samples` folder.
- TSFA source code
- Sample notebooks that illustrate:
        - How to use specific components of TSFA, with minimal config dictionaries providing required parameters. These samples can be found in the [notebooks/module_samples](notebooks/module_samples) folder.
        - Sample notebooks illustrating the various TSFA usage scenarios, emphasizing flexibility and end-to-end model training with walk-forward cross validation. These notebooks can be found in the [notebooks/how_to_use_tsfa](notebooks/how_to_use_tsfa) folder.
- [Documentation](docs) to support TSFA understanding and usage.

## Setup

### Requirements

- Python `3.10.x` - use [pyenv](https://github.com/pyenv/pyenv) for Python
  version management.
- [Poetry](https://python-poetry.org/)

### Quick Start

1. Clone the repo locally.
2. `cd` into the project
3. `poetry install` - This will leverage the poetry configuration file `pyproject.toml` and create a python virtual environment for you.
4. `poetry add <package>` - This will let you add new python dependencies from the command line
5. `poetry shell` then spins up the local virtual environment for you to develop and test code.
6. `poetry build` creates a wheel package of the `tsff` code.

## Time Series forecasting model development lifecycle

Refer to the [Time Series forecasting model development lifecycle](docs/ml_lifecycle.md) documentation for an examination of the proposed model development lifecycle and the phases TSFF can be used. However, its important to consider the following prerequisites before diving into model development:

1. The experimentation dataset should be created independent of this accelerator.
    - It is the responsibility of the Data Scientist and Engineer to ensure reconciliation and correctness of the data against source systems.
    - Source data aggregation and transformations such as imputation to cater for missing values should be done prior to using this accelerator.<br><br>

2. Exploratory Data Analysis (EDA) is conducted prior to experimentation using this accelerator.
    - It is encouraged to dedicate time for data understanding prior to experimentation. Users should have performed basic EDA steps such as data validation and missing value imputations prior to using the accelerator.
    - It is also encouraged to create an experiment hypotheses backlog such that the experiments are planned, specifying features and ML algorithms to consider.

## Time Series forecasting model development lifecycle

Refer to the [Time Series forecasting model development lifecycle](docs/ml_lifecycle.md) documentation for an examination of the proposed model development lifecycle and the phases TSFA can be used.

## Coding standards

Refer to [Time Series Forecasting Accelerator coding standards](docs/coding_standards.md) document for further details.

## Repository folder structure

The table below provides details of the folders and their content.

| Folder name | Description |
|-|-|
| docs | Accelerator documentation, and references to other forecasting resources are stored in this folder.|
| notebooks | Comprises of sample notebooks illustrating specific TSFA component usage and example notebooks to illustrate the various TSFA usage scenarios, emphasizing flexibility as well as end-to-end model training with walk forward cross validation.|
| tests | Contains unit tests for core accelerator functionality
| tsfa/common | Contains Classes that have common helper methods used for data preparation and model training. E.g. `ConfigParser` class that used to parse, get, and set configuration parameters.|
| tsfa/data_prep | Contains Class and functions for data preparation and data validation, respectively.|
| tsfa/evaluation | Evaluator class with functionality to compute model performance metrics such as WMAPE are saved in this folder.|
| tsfa/feature_engineering | This folder contains the feature engineering utility Classes. The `FeaturesUtils` Class is a "wrapper" class that orchestrates features computations based on specified configuration parameters. |
| tsfa/models | This folder contains the model utility Classes. The `MultivariateModelWrapper` and `UnivariateModelWrapper` Classes are "wrapper" classes to be used to add new models to the accelerator. |
| tsfa/ml_experiment.py | This is the ML Experiment orchestrator that utilizes the feature engineering utility, models, and evaluation Classes to train a specified model.

## Troubleshooting

Refer to [Troubleshooting](docs/troubleshooting.md) document for further details.

## Key Technologies

- [Azure DataBricks](https://azure.microsoft.com/en-us/services/databricks/)
- [Azure DevOps](https://azure.microsoft.com/en-us/services/devops/)
- [MLFlow](https://mlflow.org/)
- [Pandas API on Spark](https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/index.html)
- [Pyspark.ml package](https://spark.apache.org/docs/2.3.0/api/python/pyspark.ml.html)
- [Technical tutorial: Random Forest models with Python and Spark ML](https://www.silect.is/blog/random-forest-models-in-spark-ml/)

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com). When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

If you would like to contribute to this library, please refer to [How to contribute to Time Series Forecasting Accelerator](CONTRIBUTING.md) document for further details.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
