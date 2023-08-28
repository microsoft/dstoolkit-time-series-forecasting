"""This submodule contains an abstract univariate model class."""
from abc import abstractmethod

import pyspark.sql.functions as sf
import pyspark.sql.types as st
from pyspark.sql import DataFrame as SparkDataFrame
from typing import Callable
from tsfa.models.base_model import BaseModel


class UnivariateModelWrapper(BaseModel):
    """Wrapper for univariate time-series models.

    To add new univariate models (arima, moving average etc.),
    implement the `_fit_and_predict_single_ts` method.
    """

    def __init__(self, model_params, dataset_schema):
        """Constructor for univariate models."""
        super().__init__(model_params, dataset_schema)
        self.grain_colnames = self.dataset_schema['grain_colnames']
        self.time_colname = self.dataset_schema['time_colname']

    def fit_and_predict(self,
                        train_sdf: SparkDataFrame,
                        test_sdf: SparkDataFrame) -> SparkDataFrame:
        """
        Model fit and predict method.

        Args:
            train_sdf (SparkDataFrame): Training dataset with TS, additional input feature columns and target column.
            test_sdf (SparkDataFrame): Testing dataset with TS and additional input feature columns.

        Returns:
            forecast_sdf (SparkDataFrame): Testing dataset including the forecast.
        """
        # Concatenate train and test into a single dataframe
        train_sdf = train_sdf.withColumn("_is_train", sf.lit(1))
        test_sdf = test_sdf.withColumn("_is_train", sf.lit(0))
        if self.target_colname not in test_sdf.columns:
            test_sdf = test_sdf.withColumn(self.target_colname, sf.lit(None))
        sdf = train_sdf.union(test_sdf)

        # Train and predict
        schema = st.StructType(
            [sdf.schema[colname] for colname in sdf.columns] + [st.StructField(self.forecast_colname, st.DoubleType())]
        )
        sdf = sdf.repartition(*self.grain_colnames)
        fit_and_predict_single_ts = self._fit_and_predict_single_ts_wrapper()
        forecast_sdf = sdf.groupBy(self.grain_colnames).applyInPandas(fit_and_predict_single_ts, schema=schema)
        # Rename output column
        forecast_sdf = forecast_sdf.drop("_is_train")
        return forecast_sdf

    @abstractmethod
    def _fit_and_predict_single_ts_wrapper(self) -> Callable:
        """
        Wrapper required to define the UDF outside of package scope, otherwise it is not accessible from the workers.
        See following post for reference:
            https://community.databricks.com/s/question/0D53f00001M7cYMCAZ/modulenotfounderror-serializationerror-when-executing-over-databricksconnect

        Returns:
            _fit_and_predict_single_ts (Callable): Function object which will be parallelized via applyInPandas(). This
                method should take in a Pandas dataframe of a single time series as the input, and return a Pandas
                dataframe that include the forecast column as output.
        """
        pass
