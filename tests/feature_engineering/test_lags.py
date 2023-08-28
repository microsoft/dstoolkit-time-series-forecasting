"""Tests for the LagsFeaturizer class."""

import pytest
import numpy as np
from pyspark.sql import Row, SparkSession, DataFrame as SparkDataFrame
from pyspark.sql.functions import to_date, count
from tsfa.data_prep.data_validation import DataConnectionError
from tsfa.feature_engineering.lags import LagsFeaturizer


@pytest.fixture(scope='module')
def mock_data_train(spark: SparkSession) -> SparkDataFrame:
    """
    Create mock train data for LagsFeaturizer (will be used for fit).

    Args:
        spark (SparkSession): Fixture for local spark session.
    Returns:
        sdf (SparkDataFrame): A toy Spark dataframe to use for unit testing.
    """
    sdf_train = spark.createDataFrame([
        Row(t='2022-01-02', x1='A', y=0),
        Row(t='2022-01-09', x1='A', y=1),
        Row(t='2022-01-16', x1='A', y=2),
        Row(t='2022-01-23', x1='A', y=3),
        Row(t='2022-01-30', x1='A', y=4),
        Row(t='2022-01-02', x1='B', y=0),
        Row(t='2022-01-09', x1='B', y=1),
        Row(t='2022-01-16', x1='B', y=2),
        Row(t='2022-01-23', x1='B', y=3),
        Row(t='2022-01-30', x1='B', y=4)
    ])
    sdf_train = sdf_train.select(to_date(sdf_train.t).alias('t'), sdf_train.x1, sdf_train.y)
    return sdf_train


@pytest.fixture(scope='module')
def mock_data_test(spark: SparkSession) -> SparkDataFrame:
    """
    Create mock test data for LagsFeaturizer (will be used for transform).

    Args:
        spark (SparkSession): Fixture for local spark session.
    Returns:
        sdf (SparkDataFrame): A toy Spark dataframe to use for unit testing.
    """
    sdf_test = spark.createDataFrame([
        Row(t='2022-02-06', x1='A', y=5),
        Row(t='2022-02-13', x1='A', y=6),
        Row(t='2022-02-20', x1='A', y=7),
        Row(t='2022-02-27', x1='A', y=8),
        Row(t='2022-02-06', x1='B', y=5),
        Row(t='2022-02-13', x1='B', y=6),
        Row(t='2022-02-20', x1='B', y=7),
        Row(t='2022-02-27', x1='B', y=8)
    ])
    sdf_test = sdf_test.select(to_date(sdf_test.t).alias('t'), sdf_test.x1, sdf_test.y)
    return sdf_test


@pytest.fixture(scope='module')
def mock_data_test_bad(spark: SparkSession) -> SparkDataFrame:
    """
    Create mock "bad" test data for LagsFeaturizer (will be used for transform). This should
    cause the LagsFeaturizer to raise a ValueError since it does not connect properly to the
    padding dataframe generated during fit.

    Args:
        spark (SparkSession): Fixture for local spark session.
    Returns:
        sdf (SparkDataFrame): A toy Spark dataframe to use for unit testing.
    """
    sdf_test_bad = spark.createDataFrame([
        Row(t='2022-03-06', x1='A', y=5),
        Row(t='2022-03-13', x1='A', y=6),
        Row(t='2022-03-20', x1='A', y=7),
        Row(t='2022-03-27', x1='A', y=8),
        Row(t='2022-03-06', x1='B', y=5),
        Row(t='2022-03-13', x1='B', y=6),
        Row(t='2022-03-20', x1='B', y=7),
        Row(t='2022-03-27', x1='B', y=8)
    ])
    sdf_test_bad = sdf_test_bad.select(to_date(sdf_test_bad.t).alias('t'), sdf_test_bad.x1, sdf_test_bad.y)
    return sdf_test_bad


def test_lags_fit_trsf(mock_data_train: SparkDataFrame, mock_data_test: SparkDataFrame):
    """
    Test LagsFeaturizer output shape and transform columns.

    Args:
        mock_data_train (SparkDataFrame): Toy input train dataframe (created by fixture).
        mock_data_test (SparkDataFrame): Toy input test dataframe (created by fixture).
    """
    # Instantiate LagsFeaturizer:
    lags = LagsFeaturizer(
        grain_colnames=['x1'],
        time_colname='t',
        colnames_to_lag=['y'],
        lag_time_steps=[1, 2],
        horizon=1,
        ts_freq='W-SUN'
    )

    # Call lags fit on mock_data_train and transform on mock_data_train and mock_data_test:
    lags.fit(mock_data_train)
    sdf_train_tf = lags.transform(mock_data_train)
    sdf_test_tf = lags.transform(mock_data_test)

    # Sort output so that comparisons can be done:
    sdf_train_tf = sdf_train_tf.sort('x1', 't')
    sdf_test_tf = sdf_test_tf.sort('x1', 't')

    # Tests for output shape:
    assert sdf_train_tf.count() == 6
    assert len(sdf_train_tf.columns) == 5
    assert sdf_test_tf.count() == 8
    assert len(sdf_test_tf.columns) == 5

    # Test for output columns:
    assert all(c in sdf_train_tf.columns for c in ('y_lag1', 'y_lag2'))
    assert all(c in sdf_test_tf.columns for c in ('y_lag1', 'y_lag2'))

    # Test that date field in both train and test outputs remain unique:
    sdf_train_tf_dates = sdf_train_tf.groupBy("t").agg(count('t'))
    assert sdf_train_tf_dates.count() == 3
    assert set(sdf_train_tf_dates.toPandas()['count(t)'].values) == {2}
    sdf_test_tf_dates = sdf_test_tf.groupBy("t").agg(count('t'))
    assert sdf_test_tf_dates.count() == 4
    assert set(sdf_test_tf_dates.toPandas()['count(t)'].values) == {2}

    # Test output values for y_lag1 and y_lag2:
    correct_train_lag1 = np.array([1, 2, 3, 1, 2, 3])
    correct_train_lag2 = np.array([0, 1, 2, 0, 1, 2])
    correct_test_lag1 = np.array([4, 5, 6, 7, 4, 5, 6, 7])
    correct_test_lag2 = np.array([3, 4, 5, 6, 3, 4, 5, 6])
    assert np.array_equal(sdf_train_tf.toPandas()['y_lag1'].values, correct_train_lag1)
    assert np.array_equal(sdf_train_tf.toPandas()['y_lag2'].values, correct_train_lag2)
    assert np.array_equal(sdf_test_tf.toPandas()['y_lag1'].values, correct_test_lag1)
    assert np.array_equal(sdf_test_tf.toPandas()['y_lag2'].values, correct_test_lag2)


def test_lags_padding_check(mock_data_train: SparkDataFrame, mock_data_test_bad: SparkDataFrame):
    """
    Test LagsFeaturizer padding check - should raise a ValueError when the transform dataframe does not
    properly connect to the padding.

    Args:
        mock_data_train (SparkDataFrame): Toy input train dataframe (created by fixture).
        mock_data_test_bad (SparkDataFrame): Toy input test dataframe (created by fixture).
    """
    lags = LagsFeaturizer(
        grain_colnames=['x1'],
        time_colname='t',
        colnames_to_lag=['y'],
        lag_time_steps=[1, 2],
        horizon=1,
        ts_freq='W-SUN'
    )

    lags.fit(mock_data_train)
    with pytest.raises(DataConnectionError):
        _ = lags.transform(mock_data_test_bad)
