"""This submodule contains an abstract multivariate model class."""
from abc import abstractmethod
from pyspark.sql import DataFrame as SparkDataFrame
from tsff.models.base_model import BaseModel


class MultivariateModelWrapper(BaseModel):
    """
    Wrapper class for multivariate models.

    To add new multivariate models (e.g. lightgbm), implement the fit() and predict() methods in the inherited classes.
    """

    @abstractmethod
    def fit(self, df: SparkDataFrame):
        """
        Model fit method. This method should be fully implemented in all inherited model classes
        (e.g. random_forest.py) to take a Spark dataframe as the input, train the model, and set the value
        of self.model to the trained model. When model training is complete, this function should also
        set the value of self._is_fit to True.

        Args:
            df (SparkDataFrame): Training dataset with input features and target column.
        """
        pass

    @abstractmethod
    def predict(self, df: SparkDataFrame) -> SparkDataFrame:
        """
        Model predict method. This method should be fully implemented in all inherited model classes
        (e.g. random_forest.py) to take a Spark dataframe as the input, and use the fitted self.model to
        make forecasts. The function should return a Spark dataframe with the forecast column appended,
        and raise a ValueError exception if self._is_fit is False.

        Args:
            df (SparkDataFrame): Testing / validation dataset.
        """
        pass

    def fit_and_predict(self, train_sdf: SparkDataFrame, test_sdf: SparkDataFrame) -> SparkDataFrame:
        """
        Model fit and predict method.

        Args:
            train_sdf (SparkDataFrame): Training dataset with TS, additional input
                feature columns and target column
            test_sdf (SparkDataFrame): Testing dataset with TS and additional input
                feature columns
        Returns:
            forecast_test_sdf (SparkDataFrame): Testing dataset including the forecast
        """
        # Fit on train dataframe and estimate fitted model parameters
        self.fit(train_sdf)
        forecast_test_sdf = self.predict(test_sdf)
        return forecast_test_sdf
