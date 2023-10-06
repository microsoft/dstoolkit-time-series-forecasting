"""This is the LagsFeaturizer submodule used by features.py to compute lag features."""

import pandas as pd
from typing import List, Callable
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.types import StructType, StructField, FloatType, StringType, DateType
from functools import partial

from tsfa.data_prep.data_validation import check_date_connection, DataConnectionError


class LagsFeaturizer:
    """Lags Featurizer for time series train and validation data frames."""

    def __init__(self,
                 grain_colnames: List[str],
                 time_colname: str,
                 colnames_to_lag: List[str],
                 lag_time_steps: List[int],
                 horizon: int,
                 ts_freq: str):
        """
        Constructor for lags featurizer.

        Args:
            grain_colnames (List[str]): List of column names to group the data frame by.
            time_colname (str): Name of the time column.
            colnames_to_lag (List[str]): List of column names to create lag_time_steps for.
            lag_time_steps (List[int]): List of integers corresponding to lag columns. E.g. setting this to [1, 2, 3]
                                        and horizon to 1 will create lag3, lag4, and lag5 features for your dataset.
            horizon (int): Forecast horizon for which the predictions will be made. E.g. setting horizon = 3 and
                           lag_time_steps = [1, 2, 3] will create lag3, lag4, and lag5 features for your dataset.
            ts_freq (str): The frequency of the time series; e.g. would be 'W-SUN' if the time series is a weekly
                           series that starts on every Sunday, or 'MS' if it's a monthly series that starts on the 1st,
                           or 'D' for daily time series.
        """
        self._grain_colnames = grain_colnames
        self._time_colname = time_colname
        self._colnames_to_lag = colnames_to_lag
        self._lag_time_steps = lag_time_steps
        self._horizon = horizon
        self._ts_freq = ts_freq
        # Size of padding rows needed from training to create lags for validation
        self._padding_size = max(self._lag_time_steps) - 1 + self._horizon

        # Has the featurizer been fit on a train dataframe?
        self._is_fit = False
        # Expected output Spark DF schemas for padding dataframe and lags features
        self._create_result_schemas()
        # Spark dataframe that holds padding data for all grains
        self._padding_df = None
        # padding dataframe column names
        self._padding_df_colnames = self._padding_schema.names
        # To join lags feature columns together with the full features set
        self._join_colnames = self._grain_colnames + [self._time_colname]

    def _create_result_schemas(self):
        """Helper method to set the schema for the expected output Spark dataframe."""
        lags_features_schema = [StructField(self._time_colname, DateType())]
        padding_schema = [StructField(self._time_colname, DateType())]
        # Feature columns that will get created within this class
        self._feature_colnames = []

        for col in self._grain_colnames:
            lags_features_schema.append(StructField(col, StringType(), True))
            padding_schema.append(StructField(col, StringType(), True))

        for col in self._colnames_to_lag:
            padding_schema.append(StructField(col, FloatType(), True))
            for value in self._lag_time_steps:
                order = value - 1 + self._horizon
                lags_features_schema.append(StructField(f"{col}_lag{order}", FloatType(), True))
                # Adding to the list of features that gets created in this class
                self._feature_colnames.append(f"{col}_lag{order}")

        # Specify output schemas:
        self._lags_features_schema = StructType(lags_features_schema)
        self._padding_schema = StructType(padding_schema)

    @staticmethod
    def _fit_per_grain_wrapper() -> Callable:
        """
        Wrapper required to define the UDF outside of package scope, otherwise it is not accessible from the workers.
        See following post for reference:
            https://community.databricks.com/s/question/0D53f00001M7cYMCAZ/modulenotfounderror-serializationerror-when-executing-over-databricksconnect

        Returns:
            _fit_per_grain (Callable): Function object which will be parallelized via applyInPandas().
        """
        def _fit_per_grain(df_single_ts: pd.DataFrame,
                           time_colname: str,
                           colnames_to_lag: List[str],
                           padding_size: int) -> pd.DataFrame:
            """
            Extracts the padding dataframe for a single time series.

            Args:
                df_single_ts (pd.DataFrame): Single time series from groupby object.
            Returns:
                df_single_ts_padding (pd.DataFrame): "Padding dataframe" of length padding_size, used for creating lags
                    for future data at inference time (like a test dataset or score dataset).
            """
            # Sort and map to float:
            df_single_ts = df_single_ts.sort_values([time_colname]).reset_index(drop=True)
            for c in colnames_to_lag:
                df_single_ts[c] = df_single_ts[c].map(float)

            # Create padding from train using its last rows of length padding_size:
            df_single_ts_padding = df_single_ts.iloc[-padding_size:]
            return df_single_ts_padding

        return _fit_per_grain

    def fit(self, df: SparkDataFrame):
        """
        Fit lag featurizer on training data frame. This will result in creating a padding dataframe that
        can be used to pad the dataset that transform() will be run on. This will be stored as an
        instance attribute.

        Args:
            df (SparkDataFrame): Training dataframe used to create the padding. A crucial prerequisite
                                 for the input dataframe is that it should have a unique set of timestamps
                                 per grain at the specified granularity.

        """
        # Create padding from train data that we need for transforming inference data:
        self._padding_df = (
            df.select(self._padding_df_colnames)
              .groupBy(self._grain_colnames)
              .applyInPandas(
                partial(
                    LagsFeaturizer._fit_per_grain_wrapper(),
                    time_colname=self._time_colname,
                    colnames_to_lag=self._colnames_to_lag,
                    padding_size=self._padding_size),
                schema=self._padding_schema
            )
        )

        self._is_fit = True

    @staticmethod
    def _transform_per_grain_wrapper() -> Callable:
        """
        Wrapper required to define the UDF outside of package scope, otherwise it is not accessible from the workers.
        See following post for reference:
            https://community.databricks.com/s/question/0D53f00001M7cYMCAZ/modulenotfounderror-serializationerror-when-executing-over-databricksconnect

        Returns:
            _transform_per_grain (Callable): Function object which will be parallelized via applyInPandas().
        """
        def _transform_per_grain(df_single_ts: pd.DataFrame,
                                 time_colname: str,
                                 colnames_to_lag: List[str],
                                 lag_time_steps: List[int],
                                 padding_size: int,
                                 horizon: int) -> pd.DataFrame:
            """
            Transform series for each grain into lags features.

            Args:
                df_single_ts (pd.DataFrame): Input dataframe of a single time series to create lags for.
                time_colname (str): Name of the time column.
                colnames_to_lag (List[str]): List of column names to create lag_time_steps for.
                lag_time_steps (List[int]): List of integers corresponding to lag columns.
                padding_size (int): Size of the historical padding.
                horizon (int): Forecast horizon for which the predictions will be made.

            Returns:
                df_single_ts_result (pd.DataFrame): The resulting dataframe with lag features appended.
            """
            # Sort and map to float:
            df_single_ts = df_single_ts.sort_values([time_colname]).reset_index(drop=True)
            for c in colnames_to_lag:
                df_single_ts[c] = df_single_ts[c].map(float)

            # Create lagged data frame:
            df_list = []
            for lag in lag_time_steps:
                lag_order = lag - 1 + horizon
                df_shifted = df_single_ts[colnames_to_lag].shift(lag_order)
                df_shifted.columns = [x + "_lag" + str(lag_order) for x in df_shifted.columns]
                df_list.append(df_shifted)
            df_single_ts_lag_features = pd.concat(df_list, axis=1)

            # Add the lags feature columns to the time series:
            df_result = pd.concat([df_single_ts, df_single_ts_lag_features], axis=1)
            # Drop columns we are lagging an retain computed lag features
            df_single_ts_feats = df_result[df_result.columns[~df_result.columns.isin(colnames_to_lag)]]
            # Remove padding from result; this would be either the padding rows on the test/inference datasets,
            # or the NaN rows on the train:
            df_single_ts_feats = df_single_ts_feats[padding_size:]

            # Output:
            return df_single_ts_feats

        return _transform_per_grain

    def transform(self, df: SparkDataFrame) -> SparkDataFrame:
        """
        Computes lag features on a given Spark dataframe.

        Args:
            df (SparkDataFrame): Dataframe to create lags for. A crucial prerequisite for the dataframe
                 is that it should have a unique set of dates per grain at the specified granularity.
                 Please make sure that the time column of your data is in PySpark DateType format, not
                 other formats (like datetime or pd.timestamp).

        Returns:
            df_result (SparkDataFrame): data frame with lag features

        Raises:
            RuntimeError: When fit has not been called.
        """
        # Check for fit:
        if not self._is_fit:
            raise RuntimeError("Transform cannot be called before fit.")

        # Verify that the padding dataframe connects properly with the transform dataframe:
        self._verify_padding_df(df, self._padding_df)

        # Add padding dataframe to input (the dropDuplicates() here will remove all duplicate rows, this way
        # when padding is applied to the training data, it won't append any new rows and the dataframe remains
        # unchanged):
        df_with_padding = (
            self._padding_df
            .union(df.select(self._padding_df_colnames))
            .dropDuplicates(self._join_colnames)
        )

        df_with_padding = df_with_padding.repartition(256, self._grain_colnames)

        # Parallelize per grain:
        df_feats = (
            df_with_padding
            .groupBy(self._grain_colnames)
            .applyInPandas(
                partial(
                    LagsFeaturizer._transform_per_grain_wrapper(),
                    time_colname=self._time_colname,
                    colnames_to_lag=self._colnames_to_lag,
                    lag_time_steps=self._lag_time_steps,
                    padding_size=self._padding_size,
                    horizon=self._horizon
                ),
                schema=self._lags_features_schema
            )
        )

        # Output:
        df_result = df.join(df_feats, on=self._join_colnames, how='inner')
        return df_result

    def _verify_padding_df(self, df: SparkDataFrame, padding_df: SparkDataFrame):
        """
        Performs a data check for the padding dataframe and the transform dataframe to ensure that both dataframes
        are compatible as well as connect to each other properly. This is to prevent improper usage of the
        LagsFeaturizer, in case the padding dataframe doesn't connect to the transform dataframe by time_colname
        (e.g. if transform is called on a future DF where the dates are well after the end of the train DF).

        Args:
            df (SparkDataFrame): Dataframe to create lags for in the transform() operation.
            padding_df (SparkDataFrame): The padding dataframe used by the transform() operation.

        Raises:
            DataConnectionError: If the padding dataframe does not properly connect to the transform dataframe.
        """
        # Select the unique values from each dataframe:
        pad_time_vals = padding_df.select(self._time_colname).distinct().toPandas()
        trsf_time_vals = df.select(self._time_colname).distinct().toPandas()

        # Compute the date1 (end of padding dataframe) and date2 (start of transform dataframe):
        padding_df_last = pad_time_vals.max()[0].strftime('%Y-%m-%d')
        trsf_df_earliest = trsf_time_vals.min()[0].strftime('%Y-%m-%d')

        # Padding connection check:
        connection_check = check_date_connection(date1=padding_df_last, date2=trsf_df_earliest, freq=self._ts_freq)
        pad_times_set_diff = set(pad_time_vals[self._time_colname]) - set(trsf_time_vals[self._time_colname])
        if connection_check is False and pad_times_set_diff:
            raise DataConnectionError(
                "LagsFeaturizer transform aborted. The transform dataframe does not properly connect to the "
                f"padding dataframe. The time frequency should be {self._ts_freq}, but the transform dataframe "
                f"starts at {trsf_df_earliest} while the padding dataframe ends at {padding_df_last}. "
                f"Also, ensure that your time column ({self._time_colname}) is in Pyspark DateType format."
            )
