"""Tests for FeaturesUtils class."""

import datetime as dt
import json
import os
import pytest
from pathlib import Path
from pyspark.sql import SparkSession, types as st
from pyspark_test import assert_pyspark_df_equal
from typing import Dict, List, Tuple

from tsfa.feature_engineering.features import FeaturesUtils


@pytest.fixture(scope='session')
def mock_data() -> Tuple:
    """
    Create mock data for unit tests.

    Args:
        spark (SparkSession): Fixture for local spark session.

    Returns:
        sdf (SparkDataFrame): A toy Spark dataframe.
    """
    data = [
        {'grain': 'a', 'time': dt.datetime(2022, 2, 1), 'target': 1, 'feature': 0, 'cat1': 'A', 'cat2': 'C'},
        {'grain': 'a', 'time': dt.datetime(2022, 2, 2), 'target': 3, 'feature': 2, 'cat1': 'A', 'cat2': 'C'},
        {'grain': 'a', 'time': dt.datetime(2022, 2, 3), 'target': 5, 'feature': 4, 'cat1': 'A', 'cat2': 'C'},
        {'grain': 'a', 'time': dt.datetime(2022, 2, 4), 'target': 7, 'feature': 6, 'cat1': 'B', 'cat2': 'C'},
        {'grain': 'a', 'time': dt.datetime(2022, 2, 5), 'target': 9, 'feature': 8, 'cat1': 'B', 'cat2': 'C'},
        {'grain': 'b', 'time': dt.datetime(2022, 3, 25), 'target': 7, 'feature': 8, 'cat1': 'B', 'cat2': 'C'}
    ]
    schema = st.StructType(
        [
            st.StructField('grain', st.StringType()),
            st.StructField('time', st.DateType()),
            st.StructField('target', st.IntegerType()),
            st.StructField('feature', st.IntegerType()),
            st.StructField('cat1', st.StringType()),
            st.StructField('cat2', st.StringType()),
        ]
    )
    return data, schema


@pytest.fixture(scope='session')
def mock_data_complete() -> Tuple:
    """
    Create complete mock data for unit tests.

    Args:
        spark (SparkSession): Fixture for local spark session.

    Returns:
        sdf (SparkDataFrame): A toy Spark dataframe.
    """
    data = [
        {'grain': 'a', 'time': dt.datetime(2022, 2, 1), 'target': 1},
        {'grain': 'a', 'time': dt.datetime(2022, 2, 2), 'target': 3},
        {'grain': 'a', 'time': dt.datetime(2022, 2, 3), 'target': 5},
        {'grain': 'a', 'time': dt.datetime(2022, 2, 4), 'target': 6},
        {'grain': 'b', 'time': dt.datetime(2022, 2, 1), 'target': 7},
        {'grain': 'b', 'time': dt.datetime(2022, 2, 2), 'target': 9},
        {'grain': 'b', 'time': dt.datetime(2022, 2, 3), 'target': 7},
        {'grain': 'b', 'time': dt.datetime(2022, 2, 4), 'target': 8}
    ]
    schema = st.StructType(
        [
            st.StructField('grain', st.StringType()),
            st.StructField('time', st.DateType()),
            st.StructField('target', st.IntegerType())
        ]
    )
    return data, schema


@pytest.mark.parametrize("config_feature_engineering", [{}, {'operations': {}}])
def test_no_operations(spark: SparkSession, mock_data: Tuple, config_feature_engineering: Dict):
    """
    Test `fit_transform` method with no operations specified.

    Args:
        spark (SparkSession): Fixture for local spark session.
        mock_data (Tuple): Fixture for input dataframe as a dictionary and its schema.
        config_feature_engineering (Dict): Section of the config dictionary related to the feature engineering.
    """
    config = {
        'dataset_schema': {
            'time_colname': 'time',
            'target_colname': 'target',
            'grain_colnames': ['grain'],
            'ts_freq': 'D'
        },
        'feature_engineering': config_feature_engineering
    }
    sdf = spark.createDataFrame(*mock_data)
    feature_utils = FeaturesUtils(spark_session=spark, config=config)
    output_sdf = feature_utils.fit_transform(sdf)
    assert_pyspark_df_equal(output_sdf, sdf)


def test_all_featurizers(spark: SparkSession,
                         mock_data_complete: Tuple):
    """
    Test `fit_transform` method returns no duplicate columns.

    Args:
        spark (SparkSession): Fixture for local spark session.
        mock_data (Tuple): Fixture for input dataframe as a dictionary and its schema.
    """
    holidays_json_path = f"{Path(__file__).parent}/_holidays.json"
    config = {
        'dataset_schema': {
            'time_colname': 'time',
            'target_colname': 'target',
            'grain_colnames': ['grain'],
            'ts_freq': 'D'
        },
        'feature_engineering': {
            'feature_horizon': 2,
            'operations': {
                'lags': {
                    'colnames_to_lag': ['target'],
                    'lag_time_steps': [1],
                },
                'one_hot_encoding': {
                    'categorical_colnames': ['grain']
                },
                'basic_time_based': {
                    'feature_names': ['week_of_year']
                },
                'holidays': {
                    'holidays_json_path': holidays_json_path,
                }
            }
        }
    }

    holidays = {
        'holiday1': {
            "ds": ['2022-02-04', '2022-02-05', '2022-02-06']
        }
    }
    expected = {
        'days_to_holiday1': [-1, 0, -1, 0],
        'grain_a': [1, 1, 0, 0],
        'grain_b': [0, 0, 1, 1],
        'target_lag2': [1, 3, 7, 9],
        'week_of_year': [5, 5, 5, 5]
    }
    # Create holidays file
    with open(holidays_json_path, 'w') as f:
        f.write(json.dumps(holidays))

    sdf = spark.createDataFrame(*mock_data_complete)
    feature_utils = FeaturesUtils(spark_session=spark, config=config)
    output_sdf = feature_utils.fit_transform(sdf)
    # Remove holidays file
    os.remove(holidays_json_path)
    # Check results
    assert sdf.count() == output_sdf.count() + 4
    assert len(sdf.columns) + len(expected.items()) == len(output_sdf.columns)
    assert len(expected.items()) == len(feature_utils.feature_colnames)
    assert set(expected.keys()) == set(feature_utils.feature_colnames)
    output_sdf = output_sdf[sdf.columns + list(expected.keys())]
    output_sdf = output_sdf.sort(output_sdf.columns)
    assert sdf.columns + list(expected.keys()) == output_sdf.columns
    for k, v in expected.items():
        assert list(output_sdf.select(k).toPandas()[k]) == v


@pytest.mark.parametrize(
    'categorical_colnames,expected',
    [
        (['cat1'], {'cat1_A': [1, 1, 1, 0, 0, 0], 'cat1_B': [0, 0, 0, 1, 1, 1]}),
        (['cat1', 'cat2'], {'cat1_A': [1, 1, 1, 0, 0, 0], 'cat1_B': [0, 0, 0, 1, 1, 1], 'cat2_C': [1, 1, 1, 1, 1, 1]}),
    ]
)
def test_one_hot_encoding(
    spark: SparkSession,
    mock_data: Tuple,
    categorical_colnames: List[str],
    expected: Dict
):
    """
    Test `fit_transform` method with `one_hot_encoding` operation specified.

    Args:
        spark (SparkSession): Fixture for local spark session.
        mock_data (Tuple): Fixture for input dataframe as a dictionary and its schema.
        categorical_colnames (List[str]): List of categorical columns to apply one-hot-encoding to.
        expected (Dict): Expected additional output columns.
    """
    config = {
        'dataset_schema': {
            'time_colname': 'time',
            'target_colname': 'target',
            'grain_colnames': ['grain'],
            'ts_freq': 'D'
        },
        'feature_engineering': {
            'operations': {
                'one_hot_encoding': {
                    'categorical_colnames': categorical_colnames,
                }
            }
        }
    }
    sdf = spark.createDataFrame(*mock_data).select(categorical_colnames)
    feature_utils = FeaturesUtils(spark_session=spark, config=config)
    output_sdf = feature_utils.fit_transform(sdf)
    assert sdf.count() == output_sdf.count()
    assert len(sdf.columns) + len(expected.items()) == len(output_sdf.columns)
    output_sdf = output_sdf[sdf.columns + list(expected.keys())]
    output_sdf = output_sdf.sort(output_sdf.columns)
    assert sdf.columns + list(expected.keys()) == output_sdf.columns
    for k, v in expected.items():
        assert list(output_sdf.select(k).toPandas()[k]) == v


@pytest.mark.parametrize(
    'colnames_to_lag,lag_time_steps,expected',
    [
        (['target'], [1], {'target_lag2': [1, 3, 5]}),
        (['target'], [1, 2], {'target_lag2': [3, 5], 'target_lag3': [1, 3]}),
        (['target', 'feature'], [1], {'target_lag2': [1, 3, 5], 'feature_lag2': [0, 2, 4]}),
    ]
)
def test_lags(
    spark: SparkSession,
    mock_data: Tuple,
    colnames_to_lag: List[str],
    lag_time_steps: List[int],
    expected: Dict
):
    """
    Test `fit_transform` method with `lags` operation specified.

    Args:
        spark (SparkSession): Fixture for local spark session.
        mock_data (Tuple): Fixture for input dataframe as a dictionary and its schema.
        colnames_to_lag (List[str]): List of column names to create lag_time_steps for.
        lag_time_steps (List[int]): List of integers corresponding to lag columns.
        expected (Dict): Expected additional output columns.
    """
    config = {
        'dataset_schema': {
            'time_colname': 'time',
            'target_colname': 'target',
            'grain_colnames': ['grain'],
            'ts_freq': 'D'
        },
        'feature_engineering': {
            'feature_horizon': 2,
            'operations': {
                'lags': {
                    'colnames_to_lag': colnames_to_lag,
                    'lag_time_steps': lag_time_steps,
                }
            }
        }
    }
    sdf = spark.createDataFrame(*mock_data).select(['time', 'grain'] + colnames_to_lag)
    feature_utils = FeaturesUtils(spark_session=spark, config=config)
    output_sdf = feature_utils.fit_transform(sdf)
    assert len(sdf.columns) + len(expected.items()) == len(output_sdf.columns)
    output_sdf = output_sdf[sdf.columns + list(expected.keys())]
    output_sdf = output_sdf.sort(output_sdf.columns)
    assert sdf.columns + list(expected.keys()) == output_sdf.columns
    for k, v in expected.items():
        assert list(output_sdf.select(k).toPandas()[k]) == v


@pytest.mark.parametrize(
    'feature_names,expected',
    [
        (
            ['week_of_year', 'month_of_year', 'week_of_month'],
            {
                'week_of_year': [5, 5, 5, 5, 5, 12],
                'month_of_year': [2, 2, 2, 2, 2, 3],
                'week_of_month': [1, 1, 1, 1, 1, 4]
            }
        ),
    ]
)
def test_basic_time_based(
    spark: SparkSession,
    mock_data: Tuple,
    feature_names: List[str],
    expected: Dict
):
    """
    Test `fit_transform` method with `basic_time_based` operation specified.

    Args:
        spark (SparkSession): Fixture for local spark session.
        mock_data (Tuple): Fixture for input dataframe as a dictionary and its schema.
        feature_names (List[str]): Names of time-based features to build.
        expected (Dict): Expected additional output columns.
    """
    config = {
        'dataset_schema': {
            'time_colname': 'time',
            'target_colname': 'target',
            'grain_colnames': ['grain'],
            'ts_freq': 'D'
        },
        'feature_engineering': {
            'operations': {
                'basic_time_based': {
                    'feature_names': feature_names,
                }
            }
        }
    }
    sdf = spark.createDataFrame(*mock_data).select('time')
    feature_utils = FeaturesUtils(spark_session=spark, config=config)
    output_sdf = feature_utils.fit_transform(sdf)
    assert sdf.count() == output_sdf.count()
    assert len(sdf.columns) + len(expected.items()) == len(output_sdf.columns)
    output_sdf = output_sdf[sdf.columns + list(expected.keys())]
    output_sdf = output_sdf.sort(output_sdf.columns)
    assert sdf.columns + list(expected.keys()) == output_sdf.columns
    for k, v in expected.items():
        assert list(output_sdf.select(k).toPandas()[k]) == v


def test_holidays(spark: SparkSession, mock_data: Tuple):
    """
    Test `fit_transform` method with `holidays` operation specified.

    Args:
        spark (SparkSession): Fixture for local spark session.
        mock_data (Tuple): Fixture for input dataframe as a dictionary and its schema.
    """
    holidays_json_path = f"{Path(__file__).parent}/_holidays.json"
    config = {
        'dataset_schema': {
            'time_colname': 'time',
            'target_colname': 'target',
            'grain_colnames': ['grain'],
            'ts_freq': 'D'
        },
        'feature_engineering': {
            'operations': {
                'holidays': {
                    'holidays_json_path': holidays_json_path,
                }
            }
        }
    }
    holidays = {
        'holiday1': {
            "ds": ['2022-02-04', '2022-02-05', '2022-02-06']
        },
        'holiday2': {
            "ds": ['2022-01-28', '2022-01-29']
        },
        'holiday3': {
            "ds": ['2020-08-01', '2021-08-01', '2022-08-01', '2023-08-01']
        }
    }
    expected = {
        'days_to_holiday1': [-3, -2, -1, 0, 0, 500],
        'days_to_holiday2': [3, 4, 5, 6, 7, 500],
        'days_to_holiday3': [500, 500, 500, 500, 500, 500]
    }
    # Create holidays file
    with open(holidays_json_path, 'w') as f:
        f.write(json.dumps(holidays))
    # Run test
    sdf = spark.createDataFrame(*mock_data).select('time')
    feature_utils = FeaturesUtils(spark_session=spark, config=config)
    output_sdf = feature_utils.fit_transform(sdf)
    # Remove holidays file
    os.remove(holidays_json_path)
    # Check results
    assert sdf.count() == output_sdf.count()
    assert len(sdf.columns) + len(expected.items()) == len(output_sdf.columns)
    output_sdf = output_sdf[sdf.columns + list(expected.keys())]
    output_sdf = output_sdf.sort(output_sdf.columns)
    assert sdf.columns + list(expected.keys()) == output_sdf.columns
    for k, v in expected.items():
        assert list(output_sdf.select(k).toPandas()[k]) == v


def test_additional_features(spark: SparkSession, mock_data: Tuple):
    """
    Test `fit_transform` method with `additional_feature_colnames` specified.

    Args:
        spark (SparkSession): Fixture for local spark session.
        mock_data (Tuple): Fixture for input dataframe as a dictionary and its schema.
    """
    config = {
        'dataset_schema': {
            'time_colname': 'time',
            'target_colname': 'target',
            'grain_colnames': ['grain'],
            'ts_freq': 'D'
        },
        'feature_engineering': {
            'additional_feature_colnames': ['cat1', 'cat2']
        }
    }
    # Run test
    sdf = spark.createDataFrame(*mock_data)
    feature_utils = FeaturesUtils(spark_session=spark, config=config)
    output_sdf = feature_utils.fit_transform(sdf)
    # Check results
    assert_pyspark_df_equal(output_sdf, sdf)


def test_error_missing_feature_engineering_field(spark: SparkSession):
    """
    Test error is raised when `feature_engineering` field is missing.

    Args:
        spark (SparkSession): Fixture for local spark session.
    """
    config = {
        'dataset_schema': {
            'time_colname': 'time',
            'target_colname': 'target',
            'grain_colnames': ['grain'],
            'ts_freq': 'D'
        },
    }
    with pytest.raises(KeyError):
        _ = FeaturesUtils(spark_session=spark, config=config)


def test_error_missing_feature_horizon(spark: SparkSession, mock_data: Tuple):
    """
    Test error is raised when `feature_horizon` field is missing and required.

    Args:
        spark (SparkSession): Fixture for local spark session.
        mock_data (Tuple): Fixture for input dataframe as a dictionary and its schema.
    """
    config = {
        'dataset_schema': {
            'time_colname': 'time',
            'target_colname': 'target',
            'grain_colnames': ['grain'],
            'ts_freq': 'D'
        },
        'feature_engineering': {
            'operations': {
                'lags': {
                    'colnames_to_lag': ['target'],
                    'lag_time_steps': [1, 2]
                }
            }
        }
    }
    sdf = spark.createDataFrame(*mock_data)
    feature_utils = FeaturesUtils(spark_session=spark, config=config)
    with pytest.raises(KeyError):
        feature_utils.fit(sdf)


def test_error_non_positive_horizon(spark: SparkSession):
    """
    Test error is raised when `feature_horizon` field is non-positive.

    Args:
        spark (SparkSession): Fixture for local spark session.
    """
    config = {
        'dataset_schema': {
            'time_colname': 'time',
            'target_colname': 'target',
            'grain_colnames': ['grain'],
            'ts_freq': 'D'
        },
        'feature_engineering': {'feature_horizon': 0}
    }
    with pytest.raises(ValueError):
        _ = FeaturesUtils(spark_session=spark, config=config)
