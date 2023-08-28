"""
Metric definitions are implemented as `Evaluator` classes that inherit from `BaseEvaluator`. This implements a method to
compute the specific metric at any granularity level and defines other abstract methods that have to be implemented by
child classes to implement the specific metric logic.
"""

import numpy as np
from abc import ABC, abstractmethod
from pyspark.sql import Column, DataFrame as SparkDataFrame
from typing import List


class BaseEvaluator(ABC):
    """Base abstract class for evaluators."""

    # Epsilon used to avoid division by zero errors
    EPSILON = np.finfo(np.float64).eps

    def __init__(self, metric_colname: str):
        """
        Args:
            metric_colname: Metric output column name.
        """
        self.metric_colname = metric_colname

    def compute_metric_per_grain(
        self,
        df: SparkDataFrame,
        target_colname: str,
        forecast_colname: str,
        grain_colnames: List[str] = []
    ) -> SparkDataFrame:
        """
        Compute the metric at the specified granularity level.

        Args:
            df (SparkDataFrame): Result dataframe with actuals and forecasts at a given granularity.
            target_colname (str): Name of the target column.
            forecast_colname (str): Name of the column with the forecasted values.
            grain_colnames (List[str]): Name of the columns that define the granularity level at which the metric is
                computed. If list is empty, then a single row will be returned with a metric value for the whole
                dataframe.

        Returns:
            metric_df (SparkDataFrame): Dataframe with granularity columns and metric value per grain.
        """
        metric_expr = self.get_metric_expr(target_colname, forecast_colname)
        metric_df = df.groupBy(grain_colnames).agg(metric_expr.alias(self.metric_colname))
        return metric_df

    @abstractmethod
    def get_metric_expr(self, target_colname: str, forecast_colname: str) -> Column:
        """
        Returns expression for metric computation given actual and forecast colnames.

        Args:
            target_colname (str): Name of the target column.
            forecast_colname (str): Name of the column with the forecasted values.

        Returns:
            metric_expr (Column): Spark Column representing the formula for metric computation.
        """
        pass
