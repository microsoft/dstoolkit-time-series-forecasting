"""Tests for the RandomForestRegressorModel class."""

import pytest
from pyspark.sql import Row, SparkSession, DataFrame as SparkDataFrame
from tsfa.models import RandomForestRegressorModel
from typing import Dict


@pytest.fixture(scope='module')
def mock_data(spark: SparkSession) -> SparkDataFrame:
    """
    Create mock data for testing.

    Args:
        spark (SparkSession): Fixture for local spark session.
    Returns:
        sdf (SparkDataFrame): A toy Spark dataframe to use for testing.
    """
    sdf = spark.createDataFrame([
        Row(y=100, x1=20, x2=0, cat1="A"),
        Row(y=150, x1=50, x2=0, cat1="A"),
        Row(y=125, x1=30, x2=0, cat1="B"),
        Row(y=300, x1=25, x2=1, cat1="B"),
        Row(y=315, x1=28, x2=1, cat1="C"),
        Row(y=350, x1=55, x2=1, cat1="D")
    ])
    return sdf


@pytest.fixture(scope='module')
def mock_data_schema() -> Dict:
    """Create mock dataset schema

    Returns:
        dataset_schema (Dict): A toy dataset schema dictionary.
    """
    dataset_schema = {
        "required_columns": ['x1', 'x2', 'cat1', 'y'],
        "target_colname": 'y',
        "forecast_colname": 'y_hat'
    }
    return dataset_schema


def test_set_feature_colnames(mock_data_schema: Dict):
    """Test to ensure no duplicate columns come through."""
    # Instantiate RF Model object:
    rf = RandomForestRegressorModel(
        model_params={
            "hyperparameters": {
                "numTrees": 5,
                "maxDepth": 3
            }
        },
        dataset_schema=mock_data_schema
    )
    rf.set_feature_columns(feature_colnames=['x1', 'x1', 'x2', 'x1'])
    assert rf.feature_colnames.sort() == ['x1', 'x2'].sort()


def test_output_table_shape(mock_data: SparkDataFrame, mock_data_schema: Dict):
    """
    Test RandomForestRegressorModel output shape and prediction column.

    Args:
        mock_data (SparkDataFrame): Toy input dataframe (created by fixture).
    """
    # Instantiate RF Model object:
    rf = RandomForestRegressorModel(
        model_params={
            "hyperparameters": {
                "maxBins": 4,
                "numTrees": 5,
                "impurity": "variance",
                "maxDepth": 3,
                "featureSubsetStrategy": "auto"
            }
        },
        dataset_schema=mock_data_schema
    )

    # Set feature colnames from feature engineering output:
    rf.set_feature_columns(feature_colnames=['x1', 'x2', 'cat1'])

    # Run fit and predict on mock data:
    rf.fit(mock_data)
    out_df = rf.predict(mock_data)

    # Tests:
    assert out_df.count() == mock_data.count()
    assert len(out_df.columns) == len(mock_data.columns) + 1
    assert 'y_hat' in out_df.columns


def test_error_predict_before_fit(mock_data: SparkDataFrame, mock_data_schema: Dict):
    """
    Test that RandomForestRegressorModel raises error when predict is called before fit.

    Args:
        mock_data (SparkDataFrame): Toy input dataframe (created by fixture).
    """
    rf = RandomForestRegressorModel(
        model_params={"hyperparameters": {}},
        dataset_schema=mock_data_schema
    )

    with pytest.raises(ValueError):
        _ = rf.predict(mock_data)


def test_error_too_small_max_bins(mock_data: SparkDataFrame, mock_data_schema: Dict):
    """
    Test that RandomForestRegressorModel raises error when maxBins hyperparameter is too small.

    Args:
        mock_data (SparkDataFrame): Toy input dataframe (created by fixture).
    """
    rf = RandomForestRegressorModel(
        model_params={"hyperparameters": {"maxBins": 3}},
        dataset_schema=mock_data_schema
    )

    with pytest.raises(ValueError):
        rf.fit(mock_data)
