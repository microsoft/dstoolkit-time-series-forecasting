"""This submodule contains the SimpleExpSmoothingModel class."""

from typing import Callable

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from tsff.models import UnivariateModelWrapper


class SimpleExpSmoothingModel(UnivariateModelWrapper):
    """
    Calculate the simple exponential smoothing forecasts using train data.
    Simple exponential smoothing requires a dataset with a minimum of 2 data points.
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
            Implementation of simple exponential smoothing fit and predict method for a single TS.
            The prediction is the simple exponential smoothing, included in `model_params`.
            https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.SimpleExpSmoothing.html
            https://www.statsmodels.org/v0.10.1/examples/notebooks/generated/exponential_smoothing.html

            Args:
                df (pd.DataFrame): Dataframe to be used in `fit` and `predict` methods.
                                Train fold must be identified with `_is_train = 1`.

            Returns:
                test_df (pd.DataFrame): Testing dataframe including the forecast column.

            Raises:
                ValueError: When training fold is missing
            """
            # manage params
            model_object_params = ["initialization_method", "initial_level"]
            model_init_params = {k: v for k, v in model_hyperparams.items() if k in model_object_params}
            model_fit_params = {k: v for k, v in model_hyperparams.items() if k not in model_object_params}
            # Split train and test
            train_idx = df["_is_train"] == 1
            if train_idx.sum() == 0:
                raise ValueError("Missing train fold (rows with `_is_train = 1`).")
            train_df = df[train_idx].sort_values(time_colname)
            test_df = df[~train_idx].sort_values(time_colname).reset_index(drop=True)
            # Calculate forecast horizon
            forecast_horizon = len(test_df)
            ts = np.asarray(train_df[target_colname].astype(float))
            model = SimpleExpSmoothing(endog=ts, **model_init_params)
            forecasts = model.fit(**model_fit_params).forecast(steps=forecast_horizon)
            test_df[forecast_colname] = forecasts
            return test_df

        return _fit_and_predict_single_ts
