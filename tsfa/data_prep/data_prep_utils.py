"""
This module contains the DataPrepUtils class and is used by the configuration file creation and model training &
evaluation notebooks.
"""

import json
import pandas as pd
from datetime import datetime, date
from typing import Dict, Union
from pyspark.sql import DataFrame as SparkDataFrame, functions as sf, SparkSession
from pyspark.sql.types import StructType, StructField, DateType

from tsfa.common.dataloader import DataLoader
from tsfa.data_prep.data_validation import (
    check_df_type,
    check_dataframe_nulls,
    check_dataframe_colnames,
    check_duplicate_rows,
    validate_splits_json_config,
    check_data_leakage,
    check_date_connection,
    DataConnectionError
)
from copy import deepcopy


class DataPrepUtils:
    """Utilities class for data preparation."""

    def __init__(self, spark_session: SparkSession, config_dict: Dict, time_format: str = "%Y-%m-%d"):
        """
        Constructor for the DataPrepUtils class.

        Args:
            spark_session (SparkSession): Existing spark session.
            config_dict (Dict): The main driver configuration json.
            time_format (str): Acceptable datetime formats.
        """
        self._config = config_dict
        self.spark = spark_session
        self.loader = DataLoader(spark_session)
        self.time_format = time_format
        self.time_colname = self._config["dataset_schema"]["time_colname"]
        self.target_colname = self._config["dataset_schema"]["target_colname"]
        self.grain_colnames = self._config["dataset_schema"]["grain_colnames"]
        self.required_df_columns = self._config["dataset_schema"]["required_columns"]
        self.ts_freq = self._config["dataset_schema"]["ts_freq"]
        # Train / validation / holdout information specified as
        # key-value pairs (dict). Values: Date strings specified like "2020-03-04"
        self.holdout_timeframe, self.train_validation_timeframe = dict(), dict()
        try:
            self.holdout_timeframe = self._config["data_splitting"]["holdout_timeframe"]
        except:
            print("No holdout_timeframe specified! Will ignore holdout set during data splitting.")

        try:
            self.train_validation_timeframe = self._config["data_splitting"]["train_validation_timeframe"]
        except:
            print("No train_validation_timeframe specified. Will automatically create splits "
                  "based on other config parameters.")
        try:
            self.forecast_horizon = self._config["forecast_horizon"]
        except ValueError:
            print("Forecast_horizon must be specified!")

    def load_data(self, as_pandas: bool = False) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Load the dataset provided in the Config into a dataframe.

        Args:
            as_pandas (bool): Reads in the dataset specified in the config as pandas dataframe
                              (if True) else SparkDataFrame.
        Returns:
            df (pd.DataFrame or SparkDataFrame): The dataset read in as a Spark or Pandas dataframe.

        Raises:
            ValueError: When the JSON configuration does not specify a dataset.
        """
        # Get database table names and select columns
        db_name = self._config["dataset"]["db_name"]
        table_name = self._config["dataset"]["table_name"]

        df = self.loader.read_df_from_mount(db_name, table_name, self.required_df_columns, as_pandas)
        return df

    def validate_dataframe(self, df: Union[pd.DataFrame, SparkDataFrame]):
        """Method to run some sanity checks on the dataframe to accommodate downstream workflow

        Args:
            df (Union[pd.DataFrame, SparkDataFrame]): Spark or pandas dataframe to validate
        Raises:
            ValueError, TypeError: When validation checks fail. See data_validation.py for more details.
        """
        # Check df type
        check_df_type(df)
        # Check if dataframe has expected columns: Compare with selected columns from config
        check_dataframe_colnames(list(df.columns), self.required_df_columns)
        # check nulls
        check_dataframe_nulls(df)
        # Check for duplicate rows
        check_duplicate_rows(df, colnames=df.columns)

    def train_val_holdout_split(self, df: SparkDataFrame) -> Dict:
        """
        Computes walk forward splits of a time series into a train - validation - holdout sets
        based on cross_valdation parameters specified in the config at a choice of granularity.

        Args:
            df (SparkDataFrame): Input dataframe with multiple time series
                                (eg: Collection of time series for each sub-brand and region).

        Returns:
            train_val_holdout_splits (dict): A dictionary with "grains" as the keys (eg: "Lays XL, Atlantic") and values
                                             as a nested json of "training", "validation", and "holdout" splits.

        Raises:
            ValueError: When provided data isn't rectangular
        """
        # Run some validation checks on the loaded data
        self.validate_dataframe(df)

        # Cast target column to float
        df = self._cast_target_column_to_float(df)

        # check if time series is rectangular or ragged across all grains
        # for each timestamp, count number of grains (e.g., sub-brand, region)
        # that has that timestamp represented
        num_grains_per_timestamp = df.groupBy(self.time_colname).agg(
            sf.countDistinct(*self.grain_colnames).alias("num_grains")
        )
        # Is this count the same for all dates - this means that
        # all dates are represented for all grains (assuming there are no nulls in the data).
        # raise error if not rectangular
        if num_grains_per_timestamp.select("num_grains").distinct().count() > 1:
            raise ValueError(
                "Provided data isn't rectangular, i.e., each timestamp isn't represented",
                "for each instance of specified granularity. This need to be fixed upstream",
                "(imputed) in the data engineering stages!",
            )

        # now assume rectangular data
        timestamps_pd_df = df.select(self.time_colname).dropDuplicates().toPandas()
        return self.create_walk_forward_time_splits(timestamps_pd_df)

    def create_walk_forward_time_splits(self, timestamps_df: pd.Series) -> Dict:
        """
        Create walk forward splits as train, validation and holdout sets given a timestamps series.

        Args:
            timestamps_df (pd.DataFrame): A single column of dates in string format that we want to split into train,
                                          validation and holdout sets.

        Returns:
            wf_time_splits (Dict): Key-value pairs with keys as "holdout", "training" and "validation".
                                   Associated value is a dict (or nested dicts) of "start" and "end"
                                   timestamps for "holdout" sets and train/validation folds for each split.

        Raises:
            ValueError: If user has not specified either holdout_timeframe or forecast_horizon in the config.
        """
        # Initializing a dictionary for output time splits `wf_time_splits`
        wf_time_splits = dict()

        # Assuming dates are in string format, we first parse the dates as datetimes using time_format
        timestamps_df[self.time_colname] = pd.to_datetime(timestamps_df[self.time_colname], format=self.time_format)

        # we then sort the timestamps and set the time column as the index of the time series
        timestamps_df.sort_values(self.time_colname, inplace=True)
        timestamps_df.set_index(self.time_colname, inplace=True)

        # we will now work with the index of timestamps_pd (Lets call this ts_index)
        ts_index = timestamps_df.index

        # if holdout timeframe is provided
        if self.holdout_timeframe:
            holdout_start = datetime.strptime(self.holdout_timeframe["start"], self.time_format)
            holdout_end = datetime.strptime(self.holdout_timeframe["end"], self.time_format)

            # So lets say: holdout: {"start": "2020/01/01", "end": "2020/04/01"} but these exact
            # timestamps aren't part of the time_index in the dataset, then we need to find the
            # timestamp that is closest to this that belongs to the time_index.
            # We can use index.get_loc for this and choose the appropriate "method" parameter:
            # for "start", pick closest timestamp equal to or after "start" - use "bfill"
            # for "end", pick the closest timestamp equal to or before "end' - use "ffill"

            holdout_start_idx = ts_index.get_loc(holdout_start, method="bfill")
            holdout_end_idx = ts_index.get_loc(holdout_end, method="ffill")

            # Since holdout is specified, add this piece to time splits dictionary
            wf_time_splits["holdout"] = {
                "start": self._datetime_to_string(ts_index[holdout_start_idx]),
                "end": self._datetime_to_string(ts_index[holdout_end_idx]),
            }
        else:
            # Don't consider holdout
            holdout_end_idx = holdout_start_idx = len(ts_index)

        # if train validation timeframes are provided
        if self.train_validation_timeframe:
            train_val_start = datetime.strptime(self.train_validation_timeframe["start"], self.time_format)
            train_val_end = datetime.strptime(self.train_validation_timeframe["end"], self.time_format)

            # for "train_start", pick closest timestamp equal to or after "start" - use "bfill"
            # for "val_end", pick the closest timestamp equal to or before "end' - use "ffill"
            train_start_idx = ts_index.get_loc(train_val_start, method="bfill")
            val_end_idx = ts_index.get_loc(train_val_end, method="ffill")
        else:
            train_start_idx = 0
            val_end_idx = holdout_start_idx - 1

        # set validation start index based on forecast window size and validation end index
        val_start_idx = val_end_idx - (self.forecast_horizon - 1)
        # set train end index as 1 before validation start index
        train_end_idx = val_start_idx - 1

        # Creating first time split of train and validation sets
        wf_time_splits["training"] = {
            "split1": {
                "start": self._datetime_to_string(ts_index[train_start_idx]),
                "end": self._datetime_to_string(ts_index[train_end_idx]),
            }
        }
        wf_time_splits["validation"] = {
            "split1": {
                "start": self._datetime_to_string(ts_index[val_start_idx]),
                "end": self._datetime_to_string(ts_index[val_end_idx]),
            }
        }

        # check if user has provided cross validation parameters in the config
        if not self._config["data_splitting"]["cross_validation"]:
            print(
                "Cross validation parameters not specified in the config. Assuming ",
                "num_splits = 1, i.e., doing a single split of the data into ",
                "train - validation - holdout!",
            )
        else:
            cross_val_params = self._config["data_splitting"]["cross_validation"]
            num_splits = cross_val_params["num_splits"]
            if num_splits > 1:
                step_size = cross_val_params["rolling_origin_step_size"]

                for i in range(1, num_splits):
                    current_val_start_idx = val_start_idx - i * step_size
                    current_val_end_idx = val_end_idx - i * step_size
                    wf_time_splits["training"][f"split{i+1}"] = {
                        "start": self._datetime_to_string(ts_index[train_start_idx]),
                        "end": self._datetime_to_string(ts_index[current_val_start_idx - 1]),
                    }
                    wf_time_splits["validation"][f"split{i+1}"] = {
                        "start": self._datetime_to_string(ts_index[current_val_start_idx]),
                        "end": self._datetime_to_string(ts_index[current_val_end_idx]),
                    }
            else:
                print(
                    "Doing a single split of data into train, validation and optionally holdout. ",
                    "Ignoring rolling_origin_step_size parameter.",
                )

        return wf_time_splits

    def save_time_splits_to_datastore(self,
                                      timestamps: Dict,
                                      time_splits_file_name: str,
                                      write_folder_name: str) -> str:
        """
        Save time splits to DBFS as a dictionary.

        Args:
            timestamps (dict): The created time splits to save.
            time_splits_file_name (str): The file name to save the time splits as.
            write_folder_name (str): The folder name to save the file to.

        Returns:
            local_file_path (str): The DBFS path of the saved CV time splits file.
        """
        # Save time splits json to dbfs:/tmp:
        local_file_path = f"/dbfs/tmp/{write_folder_name}/{time_splits_file_name}"

        with open(local_file_path, "w", encoding="utf-8") as json_file:
            json.dump(timestamps, json_file, default=self._datetime_to_string, indent=4)

        return local_file_path

    def get_time_splits_from_path(self):
        """Load time_splits from a path specified in the config.

        Returns:
            time_splits (Dict): Dictionary of train - validation - holdout time splits

        Raises:
            ValueError: When `time_splits_json_path` is None, there are no time splits to retrieve
        """
        if not self.config['time_splits_json_path']:
            raise ValueError((
                "No dataset splits have been defined! Use create_walk_forward_time_splits method to split "
                "your data into train, validation, holdout by specifying data_splitting parameters in the config!"))

        with open(self.config['time_splits_json_path'], 'r') as time_splits_file:
            time_splits = json.load(time_splits_file)
        print("Opening saved time splits based on path in config!")
        return time_splits

    def validate_walk_forward_time_splits(self, wf_time_splits: Dict):
        """Get config parameters and set attributes for use in model class.

        Args:
            wf_time_splits (Dict): Dictionary of train - validation - holdout time splits
        """
        # Validate if time splits config has the right format:
        print("Validating walk forward time splits for format and leakage.")
        validate_splits_json_config(time_splits=wf_time_splits)
        # Verify time splits config don't exhibit data leakage, from train to validation and from validation to
        # holdout for each split:
        check_data_leakage(wf_time_splits)

        # Since splits are good to go, convert the date strings to datetime format
        wf_time_splits_dt = self._time_splits_to_datetime(wf_time_splits)
        return wf_time_splits_dt

    def _cast_target_column_to_float(self, df: Union[pd.DataFrame, SparkDataFrame]):
        """Method to cast target column to float type

        Args:
            df (Union[pd.DataFrame, SparkDataFrame]): Spark or pandas dataframe in which target column needs
                                                      casting to float
        Returns:
            df (Union[pd.DataFrame, SparkDataFrame]): Dataframe after casting
        """
        # Cast target column to float
        if isinstance(df, pd.DataFrame):
            df[self.target_colname] = pd.to_numeric(df[self.target_colname])
        elif isinstance(df, SparkDataFrame):
            df = df.withColumn(self.target_colname, sf.col(self.target_colname).cast("float"))
        return df

    def _datetime_to_string(self, obj: Union[date, datetime]) -> str:
        """
        Converts a datetime object to a string in a specified format.

        Arguments:
            obj (Union[date, datetime]): Input datetime object.

        Returns:
            date (str): Date in string format, i.e. converting Timestamp(2020-01-01 00:00:00)
                        to "%Y-%m-%d" format returns "2020-01-01".

        Raises:
            TypeError: If input object cannot be converted.
        """
        if isinstance(obj, (datetime, date)):
            return obj.strftime(self.time_format)
        raise TypeError(f"Can't convert {type(obj)} to date string format")

    def _time_splits_to_datetime(self, time_splits: Dict):
        """
        Helper method that transforms the date strings in the time_splits_json file into pd.Datetime format.

        Args:
            time_splits (Dict): Dict with date ranges specified for training/validation/holdout
            (from _get_cv_time_splits).
        """
        time_splits_dt = deepcopy(time_splits)
        if 'holdout' in time_splits:
            time_splits_dt['holdout'] = {k:pd.to_datetime(v) for k, v in time_splits['holdout'].items()}
        if 'training' in time_splits:
            for split in time_splits['training']:
                time_splits_dt['training'][split] = \
                    {k:pd.to_datetime(v) for k, v in time_splits['training'][split].items()}
        if 'validation' in time_splits:
            for split in time_splits['validation']:
                time_splits_dt['validation'][split] = \
                    {k:pd.to_datetime(v) for k, v in time_splits['validation'][split].items()}
        return time_splits_dt

    def make_future_dataframe(self, df: SparkDataFrame, start_time: str, end_time: str = None):
        """This is a helper method to create a future dataframe.

        Args:
            df: SparkDataFrame
            start_time (str): Start timestamp for the future timeframe eg: "2022-01-01"
            end_time (str): Start timestamp for the future timeframe eg: "2022-06-01"

        Returns:
            df_timeframe (SparkDataFrame): Prepared dataframe for the specified timeframe.

        Raises:
            ValueError: If future start date provided is before the earliest date in the dataframe provided.
                        If length of future dataset based on provided dates is greater than
                        `forecast_horizon` config parameter
            DataConnectionError: If future start date does not connect with the end of historical dataframe

        """
        # Step 0: Check if future timeframe start timestamp connects with data in df timeframe.

        # Timeframe for df
        df_timestamps_pd = df.select(self.time_colname).dropDuplicates().toPandas()
        # Assuming dates are in string format, we first parse the dates as datetimes using time_format
        df_timestamps_pd[self.time_colname] = pd.to_datetime(df_timestamps_pd[self.time_colname],
                                                             format=self.time_format)
        # start and end dates of df timeframe
        df_start, df_end = min(df_timestamps_pd[self.time_colname]), max(df_timestamps_pd[self.time_colname])

        # Future timeframe
        future_start = pd.date_range(start=start_time, periods=1, freq=self.ts_freq)

        # If future start date is before df start date, raise an error
        if future_start < df_start:
            raise ValueError("The purpose of `make_future_dataframe` is to create future dataframes for forecasting. "
                             f"Your future start: {start_time} occurs before the historical data (df) begins, "
                             f"i.e., {df_start}.")

        # If future start date is after df end date, check they connect
        if future_start > df_end and not check_date_connection(date1=df_end, date2=future_start, freq=self.ts_freq):
            raise DataConnectionError(f"Future timeframe start: {future_start} does not connect with "
                                      f"last timestamp of dataframe df: {df_end}.")

        # Step 1a: Get the range of dates for timeframe based on specified frequency.
        # `end_time` not specified: Leverage the config to create a future timeframe of size forecast_horizon.
        # `end_time` specified: Create sequence of dates and verify length doesn't exceed forecast_horizon
        if not end_time:
            print("Argument end_time is None. Generating future timeframe of size `forecast_horizon`:"
                  f"{self._config['forecast_horizon']}.")
            future_timeframe_dates = pd.date_range(start=start_time,
                                                   periods=self._config['forecast_horizon'],
                                                   freq=self.ts_freq)
        else:
            future_timeframe_dates = pd.date_range(start=start_time, end=end_time, freq=self.ts_freq)
            if len(future_timeframe_dates) > self._config['forecast_horizon']:
                raise ValueError("Length of future timeframe is larger than config specified `forecast_horizon`:"
                                 f"{self._config['forecast_horizon']}.")

        future_timeframe_dates_pd = pd.DataFrame(future_timeframe_dates, columns=[self.time_colname])
        # Step 1b: Convert dates dataframe from pandas to spark
        time_colname_schema = StructType([
            StructField(self.time_colname, DateType(), True)
        ])
        future_timeframe_dates_sp = self.spark.createDataFrame(future_timeframe_dates_pd, schema=time_colname_schema)

        # Step 2a: Extract all relevant grain information from df. While we could leverage grain_colnames
        # field from the config, we will miss additional columns that in case user brings in an externally
        # prepared dataframe
        df_grains = df.select(*self.grain_colnames).dropDuplicates()
        # Step 2b: Create new timeframe dates for each grain with a cross join
        df_grains_future_timeframe = sf.broadcast(df_grains).crossJoin(future_timeframe_dates_sp)

        # Step 3: Prepare the new dataframe by bringing in all associated feature columns for the new timeframe
        # dates. Also bring in target values from `df` for the dates that have targets while leaving other
        # target values null for the dates that don't.
        df_future_timeframe = df_grains_future_timeframe.join(df,
                                                              on=df_grains_future_timeframe.columns,
                                                              how='leftouter')
        return df_future_timeframe

    def prepare_data_for_timeframe(self, df: SparkDataFrame, timeframe: Dict) -> SparkDataFrame:
        """
        Method to either filter dataframe df based on specified timeframe [or]
        create a dataframe with required columns for a timeframe that extends outside the given df

        Args:
            df (SparkDataFrame): Source dataframe to filter from or create a new dataframe from.
            timeframe (Dict): Dictionary of `start` and end date strings
                              eg: {"start": "2022-01-01", "end": "2022-06-01"}

        Returns:
            df_timeframe (SparkDataFrame): Prepared dataframe for the specified timeframe.
        """
        # Assume rectangular data and get start and end dates for df
        # Timeframe for df
        df_timestamps_pd = df.select(self.time_colname).dropDuplicates().toPandas()
        # Assuming dates are in string format, we first parse the dates as datetimes using time_format
        df_timestamps_pd[self.time_colname] = pd.to_datetime(df_timestamps_pd[self.time_colname],
                                                             format=self.time_format)
        # start and end dates of df timeframe
        df_start, df_end = min(df_timestamps_pd[self.time_colname]), max(df_timestamps_pd[self.time_colname])

        # Convert timeframe dates to datetimes and get start and end dates
        timeframe_start_dt = pd.to_datetime(timeframe["start"], format=self.time_format)
        timeframe_end_dt = pd.to_datetime(timeframe["end"], format=self.time_format)

        # check if new_timeframe is within the df timeframe. If yes, just filter
        # the data with the slice_data_by_time method in this class
        if timeframe_start_dt >= df_start and df_end >= timeframe_end_dt:
            df_timeframe = self.slice_data_by_time(df=df, start_time=timeframe_start_dt, end_time=timeframe_end_dt)
        else:
            # New timeframe either intersects df's timeframe partially or is external to df's timeframe
            # In this scenario, we need to prepare a new dataframe for the specified new timeframe - this
            # involves extracting all existing information such as granularity from df,
            # and appending new_timeframe_dates for each grain in the new dataframe
            df_timeframe = self.make_future_dataframe(df,
                                                      start_time=timeframe_start_dt,
                                                      end_time=timeframe_end_dt)

        return df_timeframe

    def slice_data_by_time(self, df: SparkDataFrame, start_time: str = None, end_time: str = None) -> SparkDataFrame:
        """
        Helper method to extract a slice of a dataframe based on time_column using a defined date_range.

        Args:
            df (SparkDataFrame): Source dataframe to create split from.
            start_time (str): Start time to filter on, e.g. '2015-01-01'. This date is included if relevant.
            end_time (str): End time to filter on. e.g. '2021-05-02'. This date is not included in the slice.

        Returns:
            df_slice (SparkDataFrame): Slice of input dataframe.

        Raises:
            ValueError: Whenever start_time is after end_time
        """
        df_slice = df
        if start_time and end_time and start_time > end_time:
            raise ValueError("Start timestamp cannot be after end timestamp!")
        if start_time:
            df_slice = df_slice.filter(sf.col(self.time_colname) >= start_time)
        if end_time:
            df_slice = df_slice.filter(sf.col(self.time_colname) <= end_time)

        return df_slice

    def compose_dataset_prefix(self) -> str:
        """
        This function generates a prefix that could be used as a folder name '<mu>_<time_series>_<product hierarchy>'.

        Returns:
            dataset_prefix_str (str): The generated dataset filename prefix.
        """
        metadata = self._config["metadata"]
        dataset_prefix_str = f'{metadata["mu"]}_{metadata["time_series"]}_{metadata["product_hierarchy"]}'
        return dataset_prefix_str
