"""This is the BasicTimeFeatures submodule that is used by features.py to compute basic time-based features."""

import pandas as pd
import datetime as dt
from math import ceil
from typing import List
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession


class BasicTimeFeatures:
    """This class extracts based time based features from a given time column."""

    def __init__(self,
                 spark_session: SparkSession,
                 feature_names: List[str],
                 time_colname: str,
                 time_format: str = "%Y-%m-%d"):
        """
        Constructor for BasicTimeFeatures.

        Args:
            spark_session (SparkSession): Existing spark session
            feature_names (List[str]): The names of features specified in the configuration file. These are expected to
                map to method names implemented in this class. E.g., feature_name "week_of_year" maps to a method
                "week_of_year" in this class.
            time_colname (str): The name of the time column to compute features from.
            time_format (str): The format of the timestamp represented in the time column "time_colname" of the data.
                               (default: %Y-%m-%d, for example, 2022-01-18)
        """
        self.spark = spark_session
        self.feature_names = feature_names
        self.time_colname = time_colname
        self._time_format = time_format

    def transform(self, df: SparkDataFrame) -> SparkDataFrame:
        """
        Computes time based features for the input dataframe.

        Args:
            df (SparkDataFrame): Dataframe to compute features on.

        Returns:
            df_output (SparkDataFrame): Dataframe with time-based features computed.
        """
        df_input = df.select(self.time_colname).distinct().toPandas()
        # Convert time_colname to datetime format:
        df_input[self.time_colname] = pd.to_datetime(df_input[self.time_colname], format=self._time_format)

        # Feature columns we create
        self._feature_colnames = []
        for feature in self.feature_names:
            compute_function = getattr(self, feature)
            df_input = compute_function(df_input)
            self._feature_colnames.append(feature)
        df_feats = self.spark.createDataFrame(df_input, schema=df_input.columns.tolist())
        # Merge results into input dataframe:
        df_result = df.join(df_feats, on=self.time_colname, how='left')

        return df_result

    def week_of_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add week of year column to a dataframe based on datetime column.

        Args:
            df (pd.DataFrame): Dataframe with time series column.

        Returns:
            df (pd.DataFrame): Dataframe with "week_of_year" column of dtype int.
        """
        df["week_of_year"] = (df[self.time_colname].dt.isocalendar().week).astype(int)
        return df

    def month_of_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add month of year column to a dataframe based on datetime column.

        Args:
            df (pd.DataFrame): Dataframe with time series column.

        Returns:
            df (pd.DataFrame): Dataframe with "month_of_year" column of dtype int.
        """
        df["month_of_year"] = (df[self.time_colname].dt.month).astype(int)
        return df

    def week_of_month(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add week of month column to a dataframe based on datetime column.

        Args:
            df (pd.DataFrame): Dataframe with time series column.

        Returns:
            df (pd.DataFrame): Dataframe with "week_of_month" column of dtype int.
        """
        # Sub-function to compute week_of_month:
        def _week_of_month(date_time: dt.datetime) -> int:
            """
            Return week of month for a datetime value.

            Args:
                date_time (dt.datetime): A single datetime value.

            Returns:
                int: Integer value between 1 and 5 for the week of month.
            """
            # Get first day of the month and day of month:
            first_day = date_time.replace(day=1)
            day_of_month = date_time.day
            # Add first day's week day and divide by 7:
            adjusted_day_of_month = day_of_month + first_day.weekday()
            week_of_month = int(ceil(adjusted_day_of_month / 7.0))
            return week_of_month

        df["week_of_month"] = df[self.time_colname].apply(lambda x: _week_of_month(x))
        return df
