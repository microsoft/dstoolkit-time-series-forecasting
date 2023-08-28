"""This submodule contains the RollingMeanModel class."""

import numpy as np
import pandas as pd
from typing import Callable

from tsfa.models import UnivariateModelWrapper


class RollingMeanModel(UnivariateModelWrapper):
    """
    Calculate a univariate mean on the last n records of the training set to be used as the prediction in each test
    record.
    """

    def _fit_and_predict_single_ts_wrapper(self) -> Callable:
        """Implementation of superclass abstract method."""
        # Capture class instance attributes used in the inner function
        model_hyperparams = self.model_hyperparams
        time_colname = self.time_colname
        target_colname = self.target_colname
        forecast_colname = self.forecast_colname

        # Define UDF
        def _fit_and_predict_single_ts(df: pd.DataFrame) -> pd.DataFrame:
            """
            Implementation of rolling mean fit and predict method for a single TS. The prediction is the average of the
            last n values, where n is the `window_size` included in `model_params`.

            Args:
                df (pd.DataFrame): Dataframe to be used in `fit` and `predict` methods. Train fold must be identified
                    with `_is_train = 1`.

            Returns:
                test_df (pd.DataFrame): Testing dataframe including the forecast column.

            Raises:
                ValueError: When training fold is missing
            """
            # Split train and test
            train_idx = df["_is_train"] == 1
            if train_idx.sum() == 0:
                raise ValueError("Missing train fold (rows with `_is_train = 1`).")
            train_df = df[train_idx].sort_values(time_colname)
            test_df = df[~train_idx].sort_values(time_colname).reset_index(drop=True)
            # Calculate average
            window_size = model_hyperparams["window_size"]
            if len(train_df) >= window_size:
                average = np.mean(train_df[target_colname][-window_size:])
            else:
                average = np.mean(train_df[target_colname])
            # Use average as prediction
            test_df[forecast_colname] = average
            return test_df

        return _fit_and_predict_single_ts
