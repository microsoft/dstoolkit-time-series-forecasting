"""
This submodule contains the SKLearnRegressorWrapper class. It supports several model types from scikit-learn.
When using the SKLearnRegressorWrapper, the config should have the "algorithm" parameter set to one of the
supported model classes, such as "SklRandomForestRegressorModel" or "SklLinearRegressionModel". Note the "Skl"
prefix - this is to distinguish SKL models defined here from the Pyspark.ML.Regression models defined in
mllib_regressor.py.

Note: SKLearn requires Pandas dataframes as inputs, rather than Spark dataframes. This means that each SKLearn
model requires a conversion of the input dataframe to Pandas, which can be time consuming if not impossible for
larger datasets. Therefore, it is recommended to only use SKLearn regressors for smaller experimentation tasks!
"""

import pandas as pd
from typing import Dict
from abc import abstractmethod
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from sklearn.pipeline import Pipeline as SklPipeline

from tsfa.models import MultivariateModelWrapper

# Supported scikit-learn regressors:
from sklearn.linear_model import LinearRegression as SklLinearRegression
from sklearn.linear_model import TweedieRegressor as SklTweedieRegressor
from sklearn.ensemble import ExtraTreesRegressor as SklExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor as SklRandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor as SklGradientBoostingRegressor
from sklearn.linear_model import ElasticNet as SklElasticNet
from sklearn.linear_model import SGDRegressor as SklSGDRegressor
from sklearn.svm import SVR as SklSVR
from sklearn.linear_model import BayesianRidge as SklBayesianRidge
from sklearn.kernel_ridge import KernelRidge as SklKernelRidge
from xgboost.sklearn import XGBRegressor as SklXGBRegressor
from lightgbm import LGBMRegressor as SklLGBMRegressor


class SKLearnRegressorWrapper(MultivariateModelWrapper):
    """Scikit-Learn Regressor model object, contains several possible supported regressors from SKLearn."""

    def __init__(self, model_params: Dict, dataset_schema: Dict):
        """
        Constructor for SKLearnRegressorWrapper class.

        Args:
            model_params (Dict): A dictionary containing the default hyperparameters of the model.
            dataset_schema (Dict): A dictionary consisting of dataset schema parameters from the driver
                                   configuration file.
        """
        super().__init__(model_params, dataset_schema)
        self._set_skl_model()
        self.spark = SparkSession.getActiveSession()

    @abstractmethod
    def _set_skl_model(self):
        """Sets the skl_model attribute with the specific scikit-learn model object."""
        pass

    def get_model(self):
        """
        Returns the SKLearn regressor model object.

        Returns:
            self.skl_model: The SKLearn Regressor model instance.
        """
        return self.skl_model

    def fit(self, df: SparkDataFrame):
        """
        Model fit method.

        Args:
            df (SparkDataFrame): Training dataset with input feature columns and target column.

        Raises:
            ValueError: When feature columns is None.
        """
        # Make sure you have input feature columns to work with:
        if not self.feature_colnames:
            raise ValueError("Run set_feature_columns() with columns you would like to include as input features.")

        # Prepare data for training - sklearn is Pandas based:
        if isinstance(df, pd.DataFrame):
            pd_df = df
        elif isinstance(df, SparkDataFrame):
            pd_df = df.toPandas()

        df_x = pd_df[self.feature_colnames]
        df_y = pd_df[self.target_colname]

        # Create training pipeline - To be extended with optional preprocessing steps:
        reg_pipeline = SklPipeline(steps=[('regressor', self.skl_model)])

        # Fit model on features:
        self.model_pipeline = reg_pipeline.fit(df_x, df_y)
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
            raise ValueError("Model fit() should be called before model predict()!")

        # Prepare data for prediction - sklearn is Pandas based:
        if isinstance(df, pd.DataFrame):
            pd_df = df
        else:
            pd_df = df.toPandas()

        df_x = pd_df[self.feature_colnames]

        # Get prediction results:
        preds = self.model_pipeline.predict(df_x)
        pd_df[self.forecast_colname] = preds

        return self.spark.createDataFrame(pd_df)

    def __str__(self):
        """
        Magic method to allow print() to print out some model details.

        Returns:
            return_str (str): Printout of some model instance details.
        """
        return_str = f"SKLearnRegressorWrapper object of the {self.model_params['algorithm']} type.\n" \
                     f"Model hyperparameters: {self.model_params['hyperparameters']}"
        return return_str


class SklLinearRegressionModel(SKLearnRegressorWrapper):
    """SklLinearRegressionModel class."""

    def _set_skl_model(self):
        """Sets the skl_model attribute with the specific SKLearn model object."""
        self.skl_model = SklLinearRegression(**self.model_hyperparams)


class SklTweedieRegressorModel(SKLearnRegressorWrapper):
    """SklTweedieRegressorModel class."""

    def _set_skl_model(self):
        """Sets the skl_model attribute with the specific SKLearn model object."""
        self.skl_model = SklTweedieRegressor(**self.model_hyperparams)


class SklExtraTreesRegressorModel(SKLearnRegressorWrapper):
    """SklExtraTreesRegressorModel class."""

    def _set_skl_model(self):
        """Sets the skl_model attribute with the specific SKLearn model object."""
        self.skl_model = SklExtraTreesRegressor(**self.model_hyperparams)


class SklRandomForestRegressorModel(SKLearnRegressorWrapper):
    """SklRandomForestRegressorModel class."""

    def _set_skl_model(self):
        """Sets the skl_model attribute with the specific SKLearn model object."""
        self.skl_model = SklRandomForestRegressor(**self.model_hyperparams)


class SklGradientBoostingRegressorModel(SKLearnRegressorWrapper):
    """SklGradientBoostingRegressorModel class."""

    def _set_skl_model(self):
        """Sets the skl_model attribute with the specific SKLearn model object."""
        self.skl_model = SklGradientBoostingRegressor(**self.model_hyperparams)


class SklElasticNetModel(SKLearnRegressorWrapper):
    """SklElasticNetModel class."""

    def _set_skl_model(self):
        """Sets the skl_model attribute with the specific SKLearn model object."""
        self.skl_model = SklElasticNet(**self.model_hyperparams)


class SklSGDRegressorModel(SKLearnRegressorWrapper):
    """SklSGDRegressorModel class."""

    def _set_skl_model(self):
        """Sets the skl_model attribute with the specific SKLearn model object."""
        self.skl_model = SklSGDRegressor(**self.model_hyperparams)


class SklSVRModel(SKLearnRegressorWrapper):
    """SklSVRModel class."""

    def _set_skl_model(self):
        """Sets the skl_model attribute with the specific SKLearn model object."""
        self.skl_model = SklSVR(**self.model_hyperparams)


class SklBayesianRidgeModel(SKLearnRegressorWrapper):
    """SklBayesianRidgeModel class."""

    def _set_skl_model(self):
        """Sets the skl_model attribute with the specific SKLearn model object."""
        self.skl_model = SklBayesianRidge(**self.model_hyperparams)


class SklKernelRidgeModel(SKLearnRegressorWrapper):
    """SklKernelRidgeModel class."""

    def _set_skl_model(self):
        """Sets the skl_model attribute with the specific SKLearn model object."""
        self.skl_model = SklKernelRidge(**self.model_hyperparams)


class SklXGBRegressorModel(SKLearnRegressorWrapper):
    """SklXGBRegressorModel class."""

    def _set_skl_model(self):
        """Sets the skl_model attribute with the specific SKLearn model object."""
        self.skl_model = SklXGBRegressor(**self.model_hyperparams)


class SklLGBMRegressorModel(SKLearnRegressorWrapper):
    """SklLGBMRegressorModel class."""

    def _set_skl_model(self):
        """Sets the skl_model attribute with the specific SKLearn model object."""
        self.skl_model = SklLGBMRegressor(**self.model_hyperparams)
