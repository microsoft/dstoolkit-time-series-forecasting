from pyspark.sql import DataFrame as SparkDataFrame
from typing import List

from tsfa.evaluation import BaseEvaluator


class CompoundEvaluator:
    """Class to manages multiple evaluators at once."""

    def __init__(self, evaluators: List[BaseEvaluator]):
        """
        Args:
            evaluators (List[BaseEvaluator]): List of evaluators to be used when computing the metrics.

        Raises:
            ValueError: When no evaluators are specified. When there are repeated metric column names.
        """
        # Check evaluators argument
        if len(evaluators) == 0:
            raise ValueError("At least one evaluator is expected")
        evaluators_metric_colnames = [evaluator.metric_colname for evaluator in evaluators]
        if len(evaluators_metric_colnames) > len(set(evaluators_metric_colnames)):
            raise ValueError('There are repeated `metric_colname` values within the evaluators.')
        # Initialize attributes
        self.evaluators = evaluators

    def compute_metric_per_grain(
        self,
        df: SparkDataFrame,
        target_colname: str,
        forecast_colname: str,
        grain_colnames: List[str] = []
    ) -> SparkDataFrame:
        """
        Compute the metric for each evaluator at the specified granularity level.

        Args:
            df (SparkDataFrame): Result dataframe with actuals and forecasts at a given granularity.
            target_colname (str): Name of the target column.
            forecast_colname (str): Name of the column with the forecasted values.
            grain_colnames (List[str]): Name of the columns that define the granularity level at which the metric is
                computed. If list is empty, then a single row will be returned with a metric value for the whole
                dataframe.

        Returns:
            metric_df (SparkDataFrame): Dataframe with granularity columns and metrics value per grain.
        """
        # Collect all metric expressions
        metric_exprs = [
            evaluator.get_metric_expr(target_colname, forecast_colname).alias(evaluator.metric_colname)
            for evaluator in self.evaluators
        ]
        # Run metrics
        metric_df = df.groupBy(grain_colnames).agg(*metric_exprs)
        return metric_df
