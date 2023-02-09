"""
This submodule contains the MLLibRegressorWrapper class. It supports several model types from
pyspark.ml.regression. When using the MLLibRegressorWrapper, the config should have the "algorithm"
parameter set to one of the supported model classes, such as "RandomForestRegressorModel"
or "LinearRegressionModel".
"""

from typing import Dict
from abc import abstractmethod
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

from tsff.models import MultivariateModelWrapper

# Supported Spark MLlib regressors:
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import FMRegressor
from pyspark.ml.regression import IsotonicRegression


class MLLibRegressorWrapper(MultivariateModelWrapper):
    """MLLib Regressor model object, contains several possible supported regressors."""

    def __init__(self, model_params: Dict, dataset_schema: Dict):
        """
        Constructor for MLLibRegressorWrapper class.

        Args:
            model_params (Dict): A dictionary containing the default hyperparameters of the model.
            dataset_schema (Dict): A dictionary consisting of dataset schema parameters from the driver
                                   configuration file.
        """
        super().__init__(model_params, dataset_schema)

        # Get one-hot-encoding option from model hyperparameters
        self._one_hot_encoding = self.model_hyperparams.get("one_hot_encoding", True)

        # Instantiate regressor
        mlreg_class = self.get_mlreg_class()
        model_hyperparams = {k: v for k, v in self.model_hyperparams.items() if k not in ["one_hot_encoding"]}
        self.mlreg_model = mlreg_class(**model_hyperparams)

    @abstractmethod
    def get_mlreg_class(self):
        """Get the specific pyspark.ml.regression class."""
        pass

    def get_model(self):
        """
        Returns the MLLib regressor model object.

        Returns:
            self.mlreg_model: The MLLib Regressor model instance.
        """
        return self.mlreg_model

    def fit(self, df: SparkDataFrame):
        """
        Model fit method.

        Args:
            df (SparkDataFrame): Training dataset with input feature columns and target column.

        Raises:
            ValueError: When feature columns is None.
        """
        # Make sure you have input feature columns to work with
        if not self.feature_colnames:
            raise ValueError("Run set_feature_columns() with columns you would like to include as input features.")

        # Check model hyperparameters
        self._check_hyperparameters(df)

        # Identify features of type string. These are not supported by MLLib regressors and require some processing
        str_feature_colnames = [
            item[0] for item in df.select(self.feature_colnames).dtypes if item[1].startswith('string')
        ]
        num_feature_colnames = [
            item[0] for item in df.select(self.feature_colnames).dtypes if not item[1].startswith('string')
        ]

        # Initialize list of pipeline stages
        stages = []

        # Feature encoding: String indexing and One-Hot encoding
        string_indexers = [
            StringIndexer(inputCol=c, outputCol=f'{c}_idx', handleInvalid='keep') for c in str_feature_colnames
        ]
        indexed_feature_colnames = [f'{c}_idx' for c in str_feature_colnames]
        stages += string_indexers

        if self._one_hot_encoding:
            one_hot_encoder = [
                OneHotEncoder(inputCol=f'{c}_idx', outputCol=f'{c}_ohe', handleInvalid='error')
                for c in str_feature_colnames
            ]
            ohe_feature_colnames = [f'{c}_ohe' for c in str_feature_colnames]
            new_feature_colnames = num_feature_colnames + ohe_feature_colnames
            stages += one_hot_encoder
        else:
            new_feature_colnames = num_feature_colnames + indexed_feature_colnames

        # Assemble specific feature columns into a concatenated vector of features using VectorAssembler. Reference:
        # https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html
        vector_assembler = VectorAssembler(inputCols=new_feature_colnames, outputCol='features', handleInvalid='skip')
        stages += [vector_assembler]

        # Define regressor object:
        self.mlreg_model.setFeaturesCol(value=vector_assembler.getOutputCol())
        self.mlreg_model.setLabelCol(value=self.target_colname)
        self.mlreg_model.setPredictionCol(value=self.forecast_colname)
        stages += [self.mlreg_model]

        # Fit model pipeline on features:
        self.model_pipeline = Pipeline(stages=stages).fit(df)
        self._is_fit = True

    def predict(self, df: SparkDataFrame) -> SparkDataFrame:
        """
        Model predict method.

        Args:
            df (SparkDataFrame): Testing / validation input features.

        Raises:
            ValueError: When predict() is run before fit().

        Returns:
            df_with_forecast (SparkDataFrame): Dataframe with predicted values for targets.
        """
        if not self._is_fit:
            raise ValueError("RF model fit() should be called before RF model predict()!")

        df_with_forecast = self.model_pipeline.transform(df)
        return df_with_forecast.select(df.columns + [self.forecast_colname])

    def _check_hyperparameters(self, df: SparkDataFrame):
        """
        Checks the values of model hyperparameters and raises an error if issues are found.

        Args:
            df (SparkDataFrame): Training dataset with input feature columns and target column.
        """
        pass

    def _check_tree_based_model_max_bins(self, df: SparkDataFrame):
        """
        Check that `maxBins` is properly set. Must be >=2 and >= number of categories for any categorical feature.

        Args:
            df (SparkDataFrame): Training dataset with input feature columns and target column.

        Raises:
            ValueError: When hyperparameter `maxBins` is not large enough.
        """
        if "maxBins" not in self.model_hyperparams:
            return
        str_feature_colnames = [
            item[0] for item in df.select(self.feature_colnames).dtypes if item[1].startswith('string')
        ]
        if len(str_feature_colnames) == 0:
            return
        cat_feat_count = max([df.select(c).dropDuplicates().count() for c in str_feature_colnames])
        if self.model_hyperparams["maxBins"] < cat_feat_count:
            raise ValueError(
                f"Hyperparameter `maxBins` must be greater than or equal to "
                f"the largest number of categories ({cat_feat_count})."
            )

    def __str__(self):
        """
        Magic method to allow print() to print out some model details.

        Returns:
            return_str (str): Printout of some model instance details.
        """
        return_str = f"MLLibRegressorWrapper object of the {self.model_params['algorithm']} type.\n" \
                     f"Model hyperparameters: {self.model_params['hyperparameters']}"
        return return_str


class RandomForestRegressorModel(MLLibRegressorWrapper):
    """RandomForestRegressorModel class."""

    def get_mlreg_class(self):
        """Get the specific pyspark.ml.regression class."""
        return RandomForestRegressor

    def _check_hyperparameters(self, df: SparkDataFrame):
        """
        Checks the values of model hyperparameters and raises an error if issues are found.

        Args:
            df (SparkDataFrame): Training dataset with input feature columns and target column.
        """
        self._check_tree_based_model_max_bins(df)


class GBTRegressorModel(MLLibRegressorWrapper):
    """GBTRegressorModel class."""

    def get_mlreg_class(self):
        """Get the specific pyspark.ml.regression class."""
        return GBTRegressor

    def _check_hyperparameters(self, df: SparkDataFrame):
        """
        Checks the values of model hyperparameters and raises an error if issues are found.

        Args:
            df (SparkDataFrame): Training dataset with input feature columns and target column.
        """
        self._check_tree_based_model_max_bins(df)


class DecisionTreeRegressorModel(MLLibRegressorWrapper):
    """DecisionTreeRegressorModel class."""

    def get_mlreg_class(self):
        """Get the specific pyspark.ml.regression class."""
        return DecisionTreeRegressor

    def _check_hyperparameters(self, df: SparkDataFrame):
        """
        Checks the values of model hyperparameters and raises an error if issues are found.

        Args:
            df (SparkDataFrame): Training dataset with input feature columns and target column.
        """
        self._check_tree_based_model_max_bins(df)


class LinearRegressionModel(MLLibRegressorWrapper):
    """LinearRegressionModel class."""

    def get_mlreg_class(self):
        """Get the specific pyspark.ml.regression class."""
        return LinearRegression


class GeneralizedLinearRegressionModel(MLLibRegressorWrapper):
    """GeneralizedLinearRegressionModel class."""

    def get_mlreg_class(self):
        """Get the specific pyspark.ml.regression class."""
        return GeneralizedLinearRegression


class FMRegressorModel(MLLibRegressorWrapper):
    """FMRegressorModel class."""

    def get_mlreg_class(self):
        """Get the specific pyspark.ml.regression class."""
        return FMRegressor


class IsotonicRegressionModel(MLLibRegressorWrapper):
    """IsotonicRegressionModel class."""

    def get_mlreg_class(self):
        """Get the specific pyspark.ml.regression class."""
        return IsotonicRegression
