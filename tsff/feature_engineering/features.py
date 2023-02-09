"""
This module contains the FeaturesUtils class, which acts as an orchestrator for additional feature engineering
submodules. The orchestrator calls the submodules and provides the main functions used in the notebooks for fit,
transform, and fit_transform.

Additional feature engineering submodules can be added to the orchestrator as needed.
"""

from typing import Dict
from tsff.feature_engineering.one_hot_encoder import OneHotEncode
from tsff.feature_engineering.lags import LagsFeaturizer
from tsff.feature_engineering.time_based import BasicTimeFeatures
from tsff.feature_engineering.holidays import DaysToHolidays
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession


class FeaturesUtils:
    """FeaturesUtils class that orchestrates feature engineering processes."""

    IMPLEMENTED_FEATURE_TYPES = {"one_hot_encoding", "lags", "basic_time_based", "holidays"}

    def __init__(self, spark_session: SparkSession, config: Dict):
        """
        Constructor for FeatureUtils class.

        Args:
            spark_session (SparkSession): Existing spark session
            config (Dict): Dictionary of config parameters which should include feature engineering parameters.

        Raises:
            ValueError: When `feature_horizon` in `config` has wrong type or value.
        """
        self.spark = spark_session
        self.time_colname = config['dataset_schema']['time_colname']
        self.target_colname = config['dataset_schema']['target_colname']
        self.grain_colnames = config['dataset_schema']['grain_colnames']
        self.ts_freq = config['dataset_schema']['ts_freq']
        self.feature_horizon = None
        self.feature_colnames = []
        if 'feature_horizon' in config['feature_engineering']:
            self.feature_horizon = config['feature_engineering']['feature_horizon']
            if type(self.feature_horizon) != int:
                raise ValueError('Argument `feature_horizon` must be of type int')
            if self.feature_horizon < 1:
                raise ValueError('Argument `feature_horizon` must be greater or equal than 1')

        self.operations = {}
        if 'operations' in config['feature_engineering']:
            self.operations = config['feature_engineering']['operations']
        self.additional_features = []
        if 'additional_feature_colnames' in config['feature_engineering']:
            self.additional_features = config['feature_engineering']['additional_feature_colnames']

        self._is_fit = False
        self._validate_config()

    def _validate_config(self):
        """
        Check if feature utils specific requirements in this class maps to config specifications basic check.

        Raises:
            ValueError: If user specifies feature_types in the config that don't have matching implementations
                (see class level variable IMPLEMENTED_FEATURE_TYPES).
        """
        config_feature_types = set(self.operations.keys())
        a_difference_b = config_feature_types.difference(self.IMPLEMENTED_FEATURE_TYPES)
        if a_difference_b:
            raise ValueError(f"Feature types {a_difference_b} is not recognized!")

    def fit(self, df: SparkDataFrame):
        """
        Method to fit the features to apply transformation for. Note: certain features will not have fit methods.
        In these scenarios, the transform methods will be used directly.

        Args:
            df (SparkDataFrame): The training dataset to use to fit feature transformations.

        Raises:
            KeyError: When `feature_horizon` field is missing in the configuration dictionary.
        """
        self._is_fit = False
        self._feature_to_module_map = dict()

        for feature_type, feature_params in self.operations.items():
            # One hot encoding:
            if feature_type == "one_hot_encoding":
                featurizer = OneHotEncode(self.spark, feature_params["categorical_colnames"])
                featurizer.fit(df)
            # Lags:
            elif feature_type == "lags":
                if self.feature_horizon is None:
                    raise KeyError('Field `feature_horizon` is missing in config')
                featurizer = LagsFeaturizer(grain_colnames=self.grain_colnames,
                                            time_colname=self.time_colname,
                                            colnames_to_lag=feature_params["colnames_to_lag"],
                                            lag_time_steps=feature_params["lag_time_steps"],
                                            horizon=self.feature_horizon,
                                            ts_freq=self.ts_freq)
                featurizer.fit(df)
            # Basic time-based features (no fit needed):
            elif feature_type == "basic_time_based":
                featurizer = BasicTimeFeatures(spark_session=self.spark,
                                               feature_names=feature_params["feature_names"],
                                               time_colname=self.time_colname)
            # Days to holidays featurizer (no fit needed):
            elif feature_type == "holidays":
                featurizer = DaysToHolidays(spark_session=self.spark,
                                            time_colname=self.time_colname,
                                            holidays_path=feature_params["holidays_json_path"])

            # Store a map of feature_type: featurizer object:
            self._feature_to_module_map[feature_type] = featurizer

        self._is_fit = True

    def transform(self, df: SparkDataFrame) -> SparkDataFrame:
        """
        This method is used to transform the columns using the fitted features.

        Args:
            df (SparkDataFrame): The dataset to use to conduct transformations.

        Returns:
            df_result (SparkDataFrame): The dataset with transformed features added.

        Raises:
            ValueError: When featurizer is fit (self._is_fit = False).
        """
        if not self._is_fit:
            raise ValueError("For featurization, fit needs to be run before transform!")

        df_result = df
        # Maintaining a set of columns for deduplication
        all_feature_colnames = set()
        for feature_type in self.operations.keys():
            # Call featurizer transform:
            featurizer = self._feature_to_module_map[feature_type]
            df_result = featurizer.transform(df_result)
            # Update output feature columns with only new columns from the featurizer
            all_feature_colnames.update(featurizer._feature_colnames)

        all_feature_colnames.update(self.additional_features)

        self.feature_colnames = list(all_feature_colnames)
        # Output final dataframe:
        return df_result

    def fit_transform(self, df: SparkDataFrame) -> SparkDataFrame:
        """
        This method combines both fit and transform methods and runs them in sequence.

        Args:
            df (SparkDataFrame): The training dataset to fit and compute feature transformations for.

        Returns:
            df_features (SparkDataFrame): Dataset with transformed features added.
        """
        # Fit on the train dataframe:
        self.fit(df)

        # Transform dataframe based on fitted object:
        df_result = self.transform(df)

        return df_result

    def __str__(self):
        """
        Magic method to prints some featurizer metadata when print() is called on a Model instance.

        Returns:
            return_str (str): Printout of some metadata values for the FeaturesUtils instance.
        """
        return_str = (
            "Feature engineering object for creating features on input data.\n"
            "The following feature engineering operations are called:\n"
            f"{list(self.operations.keys())}\n"
            f"Ready for transform?: {self._is_fit}"
        )
        return return_str
