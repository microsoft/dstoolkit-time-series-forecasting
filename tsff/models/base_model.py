"""This submodule contains an abstract univariate model class."""
from abc import ABC, abstractmethod
from typing import List, Dict
from pyspark.sql import DataFrame as SparkDataFrame


class BaseModel(ABC):
    """Abstract base model class from which univariate, multivariate wrappers and all other models will inherit."""

    def __init__(self, model_params: Dict, dataset_schema: Dict):
        """
        Args:
            model_params (Dict): A dictionary consisting of the modeling chunk of the driver configuration file.
            dataset_schema (Dict): A dictionary consisting of dataset schema parameters from the driver
                                   configuration file.
        """
        self.model_params = model_params
        self.dataset_schema = dataset_schema

        self.model_hyperparams = self.model_params['hyperparameters']
        self.target_colname = self.dataset_schema['target_colname']
        self.forecast_colname = self.dataset_schema['forecast_colname']

        self.feature_colnames = []
        self.model = None
        self._is_fit = False
        self._has_feature_columns = False

    def set_feature_columns(self, feature_colnames: List[str]):
        """
        Select which of columns get assembled as features for model training

        Args:
            feature_colnames (List): List of column names of the dataframe that we want to use for features
        """
        # Deduplicating and setting the feature column names
        self.feature_colnames = list(set(feature_colnames))
        self._has_feature_columns = True
        self._is_fit = False

    def get_feature_columns(self) -> List[str]:
        """
        Return the list of columns being used for input features in model training

        Returns:
            feature_colnames (List): List of feature columns being used

        Raises:
            ValueError: If feature columns have not been set before leveraging them in experiments
        """
        if not self._has_feature_columns:
            raise ValueError("Use set_feature_colnames() to select the list of features you "
                             "need from the data, before running get_feature_colnames()")
        return self.feature_colnames

    @abstractmethod
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
        pass
