"""
This module contains functions used for data validation, such as checking for input types,
duplicates, nulls, and data leakage.
"""

import pandas as pd
from datetime import datetime
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as sf
from typing import Dict, List, Union


def check_df_type(df: Union[pd.DataFrame, SparkDataFrame]):
    """
    Check if input dataframe is a pandas or a spark dataframe.

    Args:
        df (Union[pd.DataFrame, SparkDataFrame]): Input dataframe.

    Raises:
        TypeError: If not a pandas or a spark dataframe.
    """
    if (not isinstance(df, pd.DataFrame)) and (not isinstance(df, SparkDataFrame)):
        raise TypeError(f"Input dataframe should either be a Pandas or a spark dataframe. It is of type: {type(df)}")
    print(f"Input is of type: {type(df)}!")


def check_dataframe_nulls(df: Union[pd.DataFrame, SparkDataFrame]):
    """
    Check if nulls exist in the input dataframe.

    Args:
        df (pd.DataFrame or SparkDataFrame): Input dataframe.

    Raises:
        ValueError: If dataframe has nulls and also prints a count and percent of nulls per columns.
    """
    if isinstance(df, pd.DataFrame):
        count_nulls_per_column = df.isnull().sum()
        pct_nulls = 100 * count_nulls_per_column / len(df)
        if any(count_nulls_per_column > 0):
            print(f"Null counts per column: \n{count_nulls_per_column} \nPct. nulls per column: \n{pct_nulls}")
            raise ValueError("Dataframe has nulls!")
    elif isinstance(df, SparkDataFrame):
        # Find count for empty, None, Null, Nan with string literals.
        count_nulls_per_column = df.select(
            [
                sf.count(sf.when(sf.col(c).isNull() | sf.isnan(c), c)).alias(c)
                for c, t in df.dtypes
                if t not in ["date", "timestamp"]
            ]
        )
        row_count = df.count()
        pct_nulls = count_nulls_per_column.select(
            [(sf.col(c) * 100 / row_count).alias(f"pct_nulls_{c}") for c in count_nulls_per_column.columns]
        )
        check_nulls = count_nulls_per_column.toPandas().to_numpy()
        if check_nulls.any() > 0:
            print(f"Null counts per column: \n{count_nulls_per_column} \nPct. nulls per column: \n{pct_nulls}")
            raise ValueError("Dataframe has nulls!")
        print("Dataframe has no nulls!")


def check_dataframe_colnames(df_colnames: List[str], expected_colnames: List[str]):
    """
    Check if df has expected columns.

    Args:
        df_colnames (List[str]): List of input dataframe columns.
        expected_colnames (List[str]): List of expected columns.

    Raises:
        ValueError: If lengths of input and expected lists don't match.
        TypeError: If input arguments are not of type List.
        KeyError: If column names differ between input and expected lists.
    """
    if not isinstance(df_colnames, List):
        raise TypeError(f"{df_colnames} is not of type List")
    if not isinstance(expected_colnames, List):
        raise TypeError(f"{expected_colnames} is not of type List")
    if not len(df_colnames) == len(expected_colnames):
        raise ValueError(
            f"Lengths of column lists don't match! Found {len(df_colnames)}, expected {len(expected_colnames)}"
        )

    # Check equality of dataframe columns and expected columns
    df_colnames.sort()
    expected_colnames.sort()
    for i, col in enumerate(df_colnames):
        if col != expected_colnames[i]:
            raise KeyError(f"Dataframe columns: {df_colnames} & Expected columns: {expected_colnames} differ!")
    print("Dataframe has expected columns!")


def check_duplicate_rows(df: Union[pd.DataFrame, SparkDataFrame], colnames: List[str]):
    """
    Verify input data frame has no duplicate rows.

    Args:
        df (pd.DataFrame): Input dataframe.
        colnames (List): Columns to consider when checking for duplicate rows.

    Raises:
        ValueError: If duplicate rows are found.
    """
    if isinstance(df, pd.DataFrame):
        if len(df[df.duplicated(colnames)]) > 0:
            raise ValueError("Duplicate rows found!!")
    if isinstance(df, SparkDataFrame):
        if df.count() > df.dropDuplicates(colnames).count():
            raise ValueError("Duplicate rows found!")
    print("No duplicate rows found!")


def validate_splits_json_config(time_splits: Dict, time_format: str = "%Y-%m-%d"):
    """
    Check if time splits dictionary used for walk-forward CV is of the right format with the expected keys.

    Args:
        time_splits (Dict): Dictionary consisting of start and end timestamps for train, validation, holdout sets.
        time_format (str): Acceptable datetime formats.

    Raises:
        TypeError: If input is not of type Dict.
        KeyError: If the provided time_splits does not have the expected keys.
        ValueError: If any key of the time_splits is null or has otherwise unexpected value.
    """
    if not isinstance(time_splits, Dict):
        raise TypeError(f"time_splits should be of type Dict. Instead got `{type(time_splits)}`.")

    time_splits_keys = set(time_splits.keys())
    expected_keys = {"training", "validation"}

    if not expected_keys <= time_splits_keys:
        raise KeyError(f"Time splits is missing these expected keys: {expected_keys-time_splits_keys}")

    for key in time_splits_keys:
        if not time_splits[key]:
            raise ValueError(f"Field `{key}` in time_splits cannot be null.")
        if not isinstance(time_splits[key], Dict):
            raise TypeError(
                f"Field `{key}` in time_splits should be a dictionary. Instead got `{type(time_splits[key])}`."
            )
        if key in {"training", "validation"}:
            _validate_training_or_validation_splits(time_splits, key, time_format)
        if key == "holdout":
            _validate_holdout_splits(time_splits, key, time_format)

    print("Time splits configuration has the right format!")


def _validate_training_or_validation_splits(time_splits: Dict, key: str, time_format: str):
    """
    Check if training or validation time splits dictionary used for walk-forward CV is of the right format with the
    expected keys.

    Args:
        time_splits (Dict): Dictionary consisting of start and end timestamps for train, validation, holdout sets.
        key (str): Key of the fold being validated.
        time_format (str): Acceptable datetime formats.

    Raises:
        TypeError: If input is not of type Dict.
        KeyError: If the provided time_splits does not have the expected keys.
        ValueError: If any key of the time_splits is null or has otherwise unexpected value.
    """
    check_sub_keys = [sub_key.startswith("split") for sub_key in time_splits[key].keys()]
    if not all(check_sub_keys):
        raise ValueError(f"Field `{key}` doesn't have all expected sub fields, i.e., split1..splitk")
    check_sub_keys_dict = [isinstance(time_splits[key][sub_key], Dict) for sub_key in time_splits[key].keys()]
    if not all(check_sub_keys_dict):
        raise TypeError(f"Some expected sub fields for `{key}` aren't dictionaries.")
    check_start_end_keys = [
        True if not {"start", "end"} ^ set(time_splits[key][sub_key].keys()) else False
        for sub_key in time_splits[key].keys()
    ]
    if not all(check_start_end_keys):
        raise KeyError(f"Field `start` or `end` is missing for some splits within `{key}`.")
    for sub_key in time_splits[key].keys():
        # Check if actual timestamps are valid
        _validate_timestamps(time_splits[key][sub_key], time_format)


def _validate_holdout_splits(time_splits: Dict, key: str, time_format: str):
    """
    Check if holdout time splits dictionary used for walk-forward CV is of the right format with the expected keys.

    Args:
        time_splits (Dict): Dictionary consisting of start and end timestamps for train, validation, holdout sets.
        key (str): Key of the fold being validated.
        time_format (str): Acceptable datetime formats.

    Raises:
        TypeError: If input is not of type Dict.
        KeyError: If the provided time_splits does not have the expected keys.
        ValueError: If any key of the time_splits is null or has otherwise unexpected value.
    """
    a_b = {"start", "end"} ^ set(time_splits[key].keys())
    if len(a_b) > 0:
        raise KeyError(f"Fields in holdout timeframe differ from expected: {a_b}!")
    # Check if actual timestamps are valid
    _validate_timestamps(time_splits[key], time_format)


def _validate_timestamps(start_end_timestamps: Dict, time_format: str = "%Y-%m-%d"):
    """
    Helper method to validate start and end timestamps within time splits config.

    Args:
        start_end_timestamps (Dict): A dict object with the following format:
            {
               "start": "2020-01-01",
               "end": "2020-04-01"
            }
        time_format (str): Acceptable datetime formats (defaults to "%Y-%m-%d")

    Raises:
        ValueError: If any key of the time_splits is null or has otherwise unexpected value.
        ValueError: If the start and end timestamps are misaligned.
        TypeError: If any keys of time_splits are in an incorrect format.
    """
    for key in {"start", "end"}:
        if not start_end_timestamps[key]:
            raise ValueError(f"Timestamp value for `{key}` cannot be null!")
        if not isinstance(start_end_timestamps[key], str):
            raise TypeError(f"`{key}` should be of type `str`. Instead got `{type(start_end_timestamps[key])}`")
        try:
            timestamp = start_end_timestamps[key]
            if isinstance(timestamp, str):
                res = bool(datetime.strptime(timestamp, time_format))
            else:
                res = bool(timestamp)
        except ValueError:
            res = False
        if not res:
            raise ValueError(f"Timestamp `{timestamp}` is not of the expected format (`{time_format}`)")

    start_str = start_end_timestamps["start"]
    end_str = start_end_timestamps["end"]

    start_timestamp = datetime.strptime(start_str, time_format)
    end_timestamp = datetime.strptime(end_str, time_format)

    if end_timestamp <= start_timestamp:
        raise ValueError(f"Start (`{start_str}`) occurs after end (`{end_str}`).")


def check_data_leakage(time_splits: Dict):
    """
    Check if the data splits have been formed correctly. Splitting could be a simple split strategy:
    train-validation-holdout [or] a cross validation split strategy: walk forward splits of train &
    validation and a holdout set.

    Args:
        time_splits (Dict): Dict consisting of start and end timestamps for train, validation, holdout sets.

    Raises:
        ValueError: If data leakage is detected.
    """
    date_format = "%Y-%m-%d"
    holdout_start = None
    if "holdout" in set(time_splits.keys()) and "start" in set(time_splits["holdout"].keys()):
        holdout_start = datetime.strptime(time_splits["holdout"]["start"], date_format)

    for split, _ in time_splits["training"].items():
        train_end = datetime.strptime(time_splits["training"][split]["end"], date_format)
        val_start = datetime.strptime(time_splits["validation"][split]["start"], date_format)
        val_end = datetime.strptime(time_splits["validation"][split]["end"], date_format)

        if val_start <= train_end:
            raise ValueError(
                f"Data leakage detected! Training set for `{split}` is leaking "
                f"into validation set! train_end (`{train_end}`) > val_start (`{val_start}`)"
            )
        if holdout_start and holdout_start <= val_end:
            raise ValueError(
                f"Data leakage detected! Validation set for `{split}` is leaking into holdout set! "
                f"val_end (`{val_end}`) > holdout_start (`{holdout_start}`)"
            )

    print("No data leakage!")


def check_date_connection(date1: str, date2: str, freq: str) -> bool:
    """
    This function checks that the two input dates, date1 and date2, correspond to the given time frequency (freq).
    For example, if freq = 'W-SUN' (meaning weekly starting on Sundays), then the check will return True if date2
    is exactly 1 week after date1, and if both dates are Sundays. Similarly, if freq = 'MS' (meaning monthly
    starting on the 1st), then the check will return True if date1 and date2 are both the first of the month and
    if date2 is exactly 1 month after date1.

    This function is often used to check if two dataframes (e.g. a train and a test) connect to each other
    properly, preventing unintended gaps between the two dataframes.

    Args:
        date1 (str): The first (earlier) date to check, in "YYYY-MM-DD" format.
        date2 (str): The second (later) date to check, in "YYYY-MM-DD" format.
        freq (str): The intended time series frequency (e.g. "W-SUN").
                    This argument should match one of the provided Pandas datetime offsets:
                    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#anchored-offsets

    Returns:
        (bool): Whether date1 and date2 correspond to the given time series frequency.

    Raises:
        ValueError: If the input dates do not correspond to the freq parameter. For example, if freq = 'W-SUN'
                    but the date1 argument is not a Sunday.
    """
    date1_ts = pd.Timestamp(date1)
    date2_ts = pd.Timestamp(date2)
    date_range = pd.date_range(start=date1, periods=2, freq=freq)

    if date1_ts != date_range[0]:
        raise DataConnectionError(
            f"Date1 ({date1}) does not match freq offset ({freq}). Please make sure your time series "
            "dates correspond to timeseries_freq parameter. E.g. freq should be 'W-SUN' if your time "
            "series is weekly starting on Sundays, 'MS' if it is monthly starting on the 1st of each "
            "month, etc."
        )

    return date2_ts == date_range[1]


class DataConnectionError(Exception):
    """Data connection exception, raised when input datasets (train & test) don't properly connect."""

    pass
