"""This submodule contains the ProphetModel class."""

import json
import pandas as pd
import datetime as dt
from prophet import Prophet
from typing import Callable

from tsff.models import UnivariateModelWrapper


class ProphetModel(UnivariateModelWrapper):
    """Prophet model object."""

    def __init__(self, model_params, dataset_schema):
        """Constructor for Prophet model."""
        super().__init__(model_params, dataset_schema)
        # Create holidays DF (if specified)
        if self.model_hyperparams.get("holidays_json_path"):
            self.holidays_df = self._create_holidays_df(self.model_hyperparams["holidays_json_path"])
        else:
            self.holidays_df = None

    def _fit_and_predict_single_ts_wrapper(self) -> Callable:
        """Implementation of superclass abstract method."""
        # Capture class instance attributes used in the inner function
        model_hyperparams = self.model_hyperparams
        time_colname = self.time_colname
        target_colname = self.target_colname
        forecast_colname = self.forecast_colname
        feature_colnames = self.feature_colnames
        holidays_df = self.holidays_df

        # Define UDF
        def _fit_and_predict_single_ts(df: pd.DataFrame) -> pd.DataFrame:
            """
            Implementation of Prophet fit and predict method for a single TS.

            Args:
                df (pd.DataFrame): Dataframe to be used in `fit` and `predict` methods. Train fold must be identified
                    with `_is_train = 1`.

            Returns:
                test_df (pd.DataFrame): Testing dataframe including the forecast column.
            """
            # Variable `max_iterations` determines max iterations to convergence and is used by PyStan, a package
            # leveraged in Prophet. Based on initial experiments the value 250 seemed sufficient for the current
            # use case.
            max_iterations = 250
            # Manage model parameters
            max_iterations = model_hyperparams.get("max_iterations", max_iterations)
            hyperparams = {k: v for k, v in model_hyperparams.items() if k not in
                           {"max_iterations", "holidays_json_path"}}
            # Add holidays_df (default None if holidays parameter isn't present)
            hyperparams['holidays'] = holidays_df
            # Set uncertainty_samples to None if not specified
            # https://towardsdatascience.com/how-to-run-facebook-prophet-predict-x100-faster-cce0282ca77d
            hyperparams.setdefault("uncertainty_samples", None)
            # Rename columns as Prophet requires
            df = df.rename(columns={target_colname: "y", time_colname: "ds"})
            # Split train and test
            train_idx = df["_is_train"] == 1
            if train_idx.sum() == 0:
                raise ValueError("Missing train fold (rows with `_is_train = 1`).")
            train_df = df[train_idx]
            test_df = df[~train_idx].sort_values("ds").reset_index(drop=True)
            # Train
            model = Prophet(**hyperparams)
            for colname in feature_colnames:
                model.add_regressor(colname)

            model.fit(train_df, iter=max_iterations)
            # Remove unused objects to optimize Prophet size
            # https://github.com/facebook/prophet/issues/1159
            model.history = 0
            model.history_dates = 0
            model.stan_backend = None
            # Make predictions
            forecast_df = model.predict(test_df)
            test_df[forecast_colname] = forecast_df["yhat"]
            # Rename columns back to original names
            test_df = test_df.rename(columns={"y": target_colname, "ds": time_colname})
            return test_df

        return _fit_and_predict_single_ts

    def _create_holidays_df(self, holidays_json_path: str) -> pd.DataFrame:
        """
        This method will return a holidays Pandas dataframe given a holidays JSON file.

        Args:
            holidays_json_path (str): Path to the holidays JSON file; which should have the following format:
                                      {'easter': {'ds': ['2015-04-05', '2016-03-27', ...],
                                                  'lower_window': -7,
                                                  'upper_window': 7},
                                       'superbowl': {'ds': ['2015-02-01', '2016-02-07', ...],
                                                     'lower_window': -7,
                                                     'upper_window': 7},
                                        ...
                                       }

        Returns:
            holidays_df (pd.DataFrame): Pandas dataframe of holidays used for Prophet model holidays parameter.
        """
        # Read json holidays file:
        with open(holidays_json_path) as f:
            holidays_dict = json.load(f)

        ds_list = []
        holiday_name_list = []
        lower_window_list = []
        upper_window_list = []
        for holiday in holidays_dict:
            holiday_ds_list = [dt.datetime.strptime(x, "%Y-%m-%d") for x in holidays_dict[holiday]["ds"]]
            num_days = len(holiday_ds_list)
            holiday_name_list_tmp = [holiday] * num_days
            ds_list += holiday_ds_list
            holiday_name_list += holiday_name_list_tmp
            lower_window_list += [holidays_dict[holiday]["lower_window"]] * num_days
            upper_window_list += [holidays_dict[holiday]["upper_window"]] * num_days

        holidays_df = pd.DataFrame(
            list(zip(ds_list, holiday_name_list, lower_window_list, upper_window_list)),
            columns=["ds", "holiday", "lower_window", "upper_window"],
        )

        return holidays_df
