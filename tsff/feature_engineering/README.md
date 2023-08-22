# Feature Engineering

Module to perform Feature Engineering. This directory houses specific feature engineering submodules with the ability for users to add more.

It is built as an orchestrator class that pulls in different feature submodules as specified by the configuration. Additional parameters required for each specific transformation are also specified in the configuration file. The orchestrator class is `FeaturesUtils` (implemented in `features.py`), and all others Python files are specific implementations of different feature transformations. Each feature transformation should at least implement a `transform` method and, in most of cases, also a `fit` method.

Currently implemented feature transformations are:

- Holidays
- Lags
- One Hot Encoding
- Time based

This module also enables the users to use features that were computed externally to `tsfa`. This is done using the "additional features" functionality.

## features.py

Orchestrator class that pulls in different feature submodules as specified by the configuration. The transformations to apply are specified in the configuration and loop through in the `fit` method.

Provides methods to:

- Fit all feature transformations
- Transform given dataframe by applying all feature transformations

Once all transformations have been applied to a dataframe, all used raw column are dropped.

## holidays.py

Class that implements `transform` methods for the creation of the days-to-major-holidays feature.

This feature does not required a `fit` step. If days-to-major-holidays is a feature engineering step in your config file, make sure the config file also contains the correct path to the JSON containing the list of holiday dates.

## lags.py

Class that implements `fit` and `transform` methods for creation of lag variables.

Lags transformation has the peculiarity that require information from rows in the past to fill present rows. Therefore, some rows from the train fold are required when transforming the validation fold. In the `fit` method these rows are stored (called a "padding" dataframe). Later in the `transform` method the expected shift is applied, including the rows from the train fold. The user can also specify a custom padding dataframe if necessary, to ensure that `transform` can be used on datasets from a later date.

## one_hot_encoder.py

Class that implements One-Hot-Encoder `fit` and `transform` methods.

The implementation is a straightforward extension of the `sklearn` `OneHotEncoder` class.

## time_based.py

Class that implements Time Based `transform` methods. These are basically:

- Week of year
- Month of year
- Week of month

These features do not required any `fit` step because the are basic transformations of the Time Series time column.
