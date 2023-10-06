"""Tests for the ProphetModel class."""

import pandas as pd
import pytest
from tsfa.models import ProphetModel
from typing import Dict


@pytest.fixture(scope='module')
def mock_data() -> pd.DataFrame:
    """
    Create mock data for testing.

    Returns:
        df (pd.DataFrame): A toy Pandas dataframe to use for testing.
    """
    n = 100
    n_train = 80
    times = list(pd.date_range(start='2020-01-01', periods=n, freq='MS'))
    raw_data = dict(
        grain=['A'] * n + ['B'] * n,
        time=times + times,
        target=list(range(0, n)) + list(range(n, 0, -1)),
        _is_train=[1] * n_train + [0] * (n - n_train) + [1] * n_train + [0] * (n - n_train),
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
        "required_columns": ['grain', 'time', 'target'],
        "grain_colnames": ['grain'],
        "time_colname": 'time',
        "target_colname": 'target',
        "forecast_colname": 'forecast'
    }
    return dataset_schema


def test_output_table_shape(mock_data: pd.DataFrame, mock_data_schema: Dict):
    """
    Test ProphetModel output shape.

    Args:
        mock_data (pd.DataFrame): Toy input dataframe (created by fixture).
    """
    model_params = {
        "algorithm": "ProphetModel",
        "hyperparameters": {
            'daily_seasonality': False,
            'weekly_seasonality': False,
            'yearly_seasonality': False
        },
        "model_name_prefix": "prophet_model"
    }

    prophet_model = ProphetModel(
        model_params=model_params,
        dataset_schema=mock_data_schema
    )
    fit_and_predict_single_ts = prophet_model._fit_and_predict_single_ts_wrapper()
    prediction = fit_and_predict_single_ts(mock_data)
    assert (prediction.columns == ['grain', 'time', 'target', '_is_train', 'forecast']).all()
    assert len(prediction) == (mock_data['_is_train'] == 0).sum()


def test_error_missing_train_fold(mock_data: pd.DataFrame, mock_data_schema: Dict):
    """
    Test ProphetModel error raising.

    Args:
        mock_data (pd.DataFrame): Toy input dataframe (created by fixture).
    """
    prophet_model = ProphetModel(
        model_params={"hyperparameters": {}},
        dataset_schema=mock_data_schema
    )
    fit_and_predict_single_ts = prophet_model._fit_and_predict_single_ts_wrapper()
    with pytest.raises(ValueError):
        _ = fit_and_predict_single_ts(mock_data[mock_data["_is_train"] == 0])
