"""Tests for CompoundEvaluator class."""

import pytest
from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
from pyspark_test import assert_pyspark_df_equal

from tsff.evaluation import CompoundEvaluator, OffsetErrEvaluator, WMapeEvaluator


@pytest.fixture
def data(spark: SparkSession) -> SparkDataFrame:
    """
    Create mock data for unit tests.

    Args:
        spark (SparkSession): Fixture for local spark session.

    Returns:
        sdf (SparkDataFrame): A toy Spark dataframe.
    """
    sdf = spark.createDataFrame(
        [
            {"target": 1, "forecast": 3},
            {"target": 2, "forecast": 3},
            {"target": 4, "forecast": 3},
            {"target": 5, "forecast": 5},
        ]
    )
    return sdf


def test_compute_metric_per_grain_mutliple_evaluators(data: SparkDataFrame, spark: SparkSession):
    """
    Test method `compute_metric_per_grain` with multiple evaluators.

    Args:
        data (SparkDataFrame): Fixture for input dataframe.
        spark (SparkSession): Fixture for local spark session.
    """
    evaluator = CompoundEvaluator(
        [
            WMapeEvaluator(metric_colname="wmape1"),
            WMapeEvaluator(metric_colname="wmape2"),
            OffsetErrEvaluator(),
        ]
    )
    output_sdf = evaluator.compute_metric_per_grain(
        data,
        target_colname='target',
        forecast_colname='forecast'
    )
    expected_sdf = spark.createDataFrame([{"wmape1": 4 / 12, "wmape2": 4 / 12, "offset_err": 2 / 12}])
    # Ignore rows order when comparing
    output_sdf = output_sdf.sort(output_sdf.columns)
    expected_sdf = expected_sdf.sort(expected_sdf.columns)
    assert_pyspark_df_equal(output_sdf, expected_sdf)


def test_compute_metric_per_grain_one_evaluator(data: SparkDataFrame, spark: SparkSession):
    """
    Test method `compute_metric_per_grain` with a single evaluator.

    Args:
        data (SparkDataFrame): Fixture for input dataframe.
        spark (SparkSession): Fixture for local spark session.
    """
    evaluator = CompoundEvaluator([WMapeEvaluator()])
    output_sdf = evaluator.compute_metric_per_grain(
        data,
        target_colname='target',
        forecast_colname='forecast'
    )
    expected_sdf = spark.createDataFrame([{"wmape": 4 / 12}])
    # Ignore rows order when comparing
    output_sdf = output_sdf.sort(output_sdf.columns)
    expected_sdf = expected_sdf.sort(expected_sdf.columns)
    assert_pyspark_df_equal(output_sdf, expected_sdf)


def test_error_no_evaluators():
    """Test error is raised when `compute_metric_per_grain` receives no evaluators."""
    with pytest.raises(ValueError):
        _ = CompoundEvaluator([])


def test_error_metric_colname_collision():
    """Test error is raised when there are mutliple evaluators with the same `metric_colname`."""
    with pytest.raises(ValueError):
        _ = CompoundEvaluator([WMapeEvaluator(), WMapeEvaluator()])
