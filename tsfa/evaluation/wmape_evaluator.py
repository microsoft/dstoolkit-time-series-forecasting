import pyspark.sql.functions as sf
from pyspark.sql import Column
from typing import Optional

from tsfa.evaluation import BaseEvaluator


class WMapeEvaluator(BaseEvaluator):
    """
    Evaluator class for WMAPE (Weighted Mean Absolute Percentage Error).
    Reference: https://www.baeldung.com/cs/mape-vs-wape-vs-wmape
    """

    def __init__(self, express_as_accuracy: bool = False, metric_colname: Optional[str] = None):
        """
        Constructor to WMape Evaluator

        Args:
            express_as_accuracy (bool) : Flag that decides if the nature of metrics is accuracy / error
            metric_colname(str): Metric output column name. If None, default based on `express_as_accuracy` is used.
        """
        if metric_colname is None:
            metric_colname = "wmape" if not express_as_accuracy else "absolute_accuracy"
        super().__init__(metric_colname)
        self.return_accuracy = express_as_accuracy

    def get_metric_expr(self, target_colname: str, forecast_colname: str) -> Column:
        """
        Returns expression for WMAPE computation (as accuracy, if specified) given actual and forecast colnames.

        Args:
            target_colname (str): Name of the target column.
            forecast_colname (str): Name of the column with the forecasted values.

        Returns:
            metric_expr (Column): Spark Column representing the formula for WMAPE / Absolute accuracy computation
            based on 'return_accuracy'
        """
        wmape_num = sf.sum(sf.abs(sf.col(target_colname) - sf.col(forecast_colname)))
        wmape_denom = sf.sum(sf.abs(sf.col(target_colname)))
        wmape_denom = sf.when(wmape_denom != 0, wmape_denom).otherwise(self.EPSILON)
        metric_expr = wmape_num / wmape_denom
        return self._get_acc_expr(metric_expr) if self.return_accuracy else metric_expr

    def _get_acc_expr(self, error_metric_expr: Column) -> Column:
        """
        Returns expression for accuracy for metric computation given wmape error metrics expression.

        Args:
            error_metric_expr (Column): Spark Column representing the formula for WMAPE computation.

        Returns:
            acc_expr (Column): Spark Column representing the formula for accuracy calculation from
            error metric expression.
        """
        acc_expr = 1 - error_metric_expr
        return acc_expr
