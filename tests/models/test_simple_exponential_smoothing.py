"""Tests for SimpleExpSmoothingModel class."""

from typing import Dict

import pandas as pd
import pytest
from tsff.models import SimpleExpSmoothingModel


@pytest.fixture(scope="session")
def mock_data() -> pd.DataFrame:
    """
    Create mock data for testing.

    Returns:
        df (pd.DataFrame): A toy Pandas dataframe to use for testing.
    """
    raw_data = dict(
        grain=["A"] * 10,
        ds=list(range(0, 10)),
        target=list(range(0, 10)),
        _is_train=[1] * 8 + [0] * 2,
    )
    df = pd.DataFrame(raw_data)
    return df


@pytest.fixture(scope='module')
def mock_data_schema() -> Dict:
    """Create mock dataset schema

    Returns:
        dataset_schema (Dict): A toy dataset schema dictionary.
    """
    dataset_schema = {
        "required_columns": ['grain', 'ds', 'target'],
        "grain_colnames": [],
        "time_colname": 'ds',
        "target_colname": 'target',
        "forecast_colname": 'yhat'
    }
    return dataset_schema


def test_output_table_shape(mock_data: pd.DataFrame, mock_data_schema: Dict):
    """
    Test SimpleExpSmoothingModel output shape.

    Args:
        mock_data (pd.DataFrame): Toy input dataframe (created by fixture).
    """
    exponential_smoothing_model = SimpleExpSmoothingModel(
        model_params={"hyperparameters": dict(smoothing_level=0.3)},
        dataset_schema=mock_data_schema
    )
    fit_and_predict_single_ts = exponential_smoothing_model._fit_and_predict_single_ts_wrapper()
    prediction = fit_and_predict_single_ts(mock_data)
    assert (prediction.columns == ["grain", "ds", "target", "_is_train", "yhat"]).all()
    assert len(prediction) == (mock_data["_is_train"] == 0).sum()


def test_error_missing_train_fold(mock_data: pd.DataFrame, mock_data_schema: Dict):
    """
    Test SimpleExpSmoothingModel error raising when missing the training fold.

    Args:
        mock_data (pd.DataFrame): Toy input dataframe (created by fixture).
    """
    exponential_smoothing_model = SimpleExpSmoothingModel(
        model_params={"hyperparameters": dict(smoothing_level=0.3)},
        dataset_schema=mock_data_schema
    )
    fit_and_predict_single_ts = exponential_smoothing_model._fit_and_predict_single_ts_wrapper()
    with pytest.raises(ValueError):
        _ = fit_and_predict_single_ts(mock_data[mock_data["_is_train"] == 0])
