"""Tests for ErrorAnalysis class."""

from datetime import datetime, timedelta
import random
import pandas as pd
from tsff.error_analysis import ErrorAnalysis


def mock_data_result(
    time_identifier: str = "week",
    keys_identifier: list = ["key1", "key2", "key3"],
    target_column_name: str = "actual",
    predicted_column_name: str = "prediction",
    iteration_num: int = 5,
    seed_id: int = 81,
) -> pd.DataFrame:
    random.seed(seed_id)
    tuples = []
    cur_week = datetime.strptime("2020-01-01", "%Y-%m-%d")
    delta = timedelta(days=7)
    for keys_number in range(1000):
        nums = [str(random.randint(1, 100)) for _ in range(len(keys_identifier))]
        for iteration in range(iteration_num):
            for lookahead in range(1, 5):
                tuples.append(
                    [cur_week + delta * iteration + delta * lookahead]
                    + nums
                    + [random.randint(80, 120), random.randint(80, 120)]
                    + [lookahead, cur_week + delta * iteration, iteration]
                )

    df = pd.DataFrame(
        tuples,
        columns=[time_identifier]
        + keys_identifier
        + [
            target_column_name,
            predicted_column_name,
            "lookahead",
            "walk",
            "iteration",
        ],
    )
    return df


def test_get_rank(
    target_column_name: str = "actual",
    time_identifier: str = "week",
    keys_identifier: list = ["key1", "key2"],
    predicted_column_name="prediction",
):
    df_mock = mock_data_result(keys_identifier)

    error_analysis = ErrorAnalysis(
        target_column_name=target_column_name,
        time_identifier=time_identifier,
        keys_identifier=keys_identifier,
        predicted_column_name=predicted_column_name
    )
    df_mock[target_column_name] = 10
    df_mock[predicted_column_name] = 10
    df_metric = error_analysis.get_metric_values(df_mock, keys=keys_identifier)
    assert len(df_metric) == len(df_mock) / 2
    assert sum(df_metric.rmse) == 0.0
    assert sum(df_metric.mse) == 0.0
    assert sum(df_metric.mape) == 0.0
