"""Tests for WMapeEvaluator class."""

import pytest
from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
from pyspark_test import assert_pyspark_df_equal
from typing import Dict, List

from tsff.evaluation import WMapeEvaluator


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
            {"grain": "a", "target": 1, "forecast": 3},
            {"grain": "a", "target": 2, "forecast": 3},
            {"grain": "a", "target": 4, "forecast": 3},
            {"grain": "b", "target": 5, "forecast": 5},
            {"grain": "b", "target": 6, "forecast": 7},
        ]
    )
    return sdf


@pytest.mark.parametrize(
    "grain_colnames,express_as_accuracy,expected",
    [
        ([], False, [{"wmape": 5 / 18}]),
        (["grain"], False,[{"grain": "a", "wmape": 4 / 7}, {"grain": "b", "wmape": 1 / 11}]),
        ([], True, [{"absolute_accuracy": 1 - (5 / 18)}]),
        (
            ["grain"], True,[
                {"grain": "a", "absolute_accuracy": 1 - (4 / 7)},
                {"grain": "b", "absolute_accuracy": 1 - (1 / 11)}
            ]
        )
    ],
)
def test_compute_metric_per_grain(
    grain_colnames: List[str],
    expected: List[Dict],
    express_as_accuracy: bool,
    data: SparkDataFrame,
    spark: SparkSession,
):
    """
    Test method compute_metric_per_grain.

    Args:
        grain_colnames (List[str]): The granularity at which user would like to compute metric.
        expected (List[Dict]): Expected output rows.
        data (SparkDataFrame): Fixture for input dataframe.
        spark (SparkSession): Fixture for local spark session.
    """
    evaluator = WMapeEvaluator(express_as_accuracy)
    output_sdf = evaluator.compute_metric_per_grain(
        data,
        target_colname='target',
        forecast_colname='forecast',
        grain_colnames=grain_colnames
    )
    expected_sdf = spark.createDataFrame(expected)
    # Ignore rows order when comparing
    output_sdf = output_sdf.sort(output_sdf.columns)
    expected_sdf = expected_sdf.sort(expected_sdf.columns)
    assert_pyspark_df_equal(output_sdf, expected_sdf)
