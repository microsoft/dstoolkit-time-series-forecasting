# Data Preparation

This module assists with data preparation steps that precede model training and evaluation. Specifically, this module provides following functionality:

- Loading of raw tables into Pandas or PySpark DataFrames
- Definition and management of train-validation-test time splits
- Checking data format, content and leakage

## data_prep_utils.py

Class to perform all data preparation actions, mainly focused on loading the data and creating the train, validation, holdout (test) splits. The data source, columns and the splits required are parametrized in the configuration file, therefore this class heavily depends on the configuration dictionary. Refer to [notebooks/samples](../notebooks/samples) folder notebooks for expected dictionary schema.

Functionality is:

- Generate a prefix to be used when storing experimentation related files
- Load a table slice into a dataframe
- Define train, validation, holdout (test) splits with options to perform splitting at granular level using [walk-forward cross validation](../docs/Model%20Experimentation%20and%20Evaluation%20Recommendations.md) or simple splitting strategy (cross validation with a single split)
- Save defined splits in JSON format
- Save the dataframe once the time splits has been applied into many train-validation-test folds
- Verify the JSON file has been properly stored

## data_validation.py

Class with utilities to check the data has the right format and content. It also provides methods to validate the time splits generated in the data preparation step.

Data format checks:

- Data format: Pandas or Spark
- Column names

Data content checks:

- Nulls
- Duplicated rows

Time splits:

- JSON file with the splits has the right format and expected keys
- Helper method to validate start and end timestamps
- No data leakage from the defined time splits
