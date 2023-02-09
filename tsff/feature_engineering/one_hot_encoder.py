"""This is the OneHotEncode submodule used by features.py to compute one-hot-encoded categorical features."""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from pyspark.sql import DataFrame as SparkDataFrame
from typing import List
from pyspark.sql import SparkSession


class OneHotEncode:
    """
    One hot encoder for categorical columns of a data frame.
    https://scikit-learn.org/0.16/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    """

    def __init__(self, spark_session: SparkSession, categorical_colnames: List[str]):
        """
        Constructor for the OneHotEncode class.

        Args:
            spark_session (SparkSession): Existing spark session
            categorical_colnames (List[str]): List of categorical columns to apply one-hot-encoding to.
        """
        self.spark = spark_session
        self._encoder = OneHotEncoder(handle_unknown='ignore')
        self._categorical_colnames = categorical_colnames
        self._is_fit = False

    def fit(self, df: SparkDataFrame):
        """
        Fit OneHotEncoder for categorical columns of a data frame.

        Args:
            df (SparkDataFrame): Dataframe with categorical columns to encode.
        """
        df_input = df.select(self._categorical_colnames).distinct().toPandas()
        self._encoder.fit(df_input)
        self._is_fit = True

    def transform(self, df: SparkDataFrame, replace_colname_spaces: bool = False) -> SparkDataFrame:
        """
        Transform categorical columns using fitted encoder.

        Args:
            df (SparkDataFrame): Dataframe with categorical columns to transform using fitted encoder.
            replace_colname_spaces (bool): True/False flag for renaming columns to remove spaces with underscores.

        Returns:
            df_result (SparkDataFrame): Distinct dataframe with categorical columns and corresponding OHE vectors.

        Raises:
             RuntimeError: When transform method is called before fit.
        """
        # Check that encoder is fit:
        if not self._is_fit:
            raise RuntimeError("Cannot call transform() method before fit().")

        # Update _feature_colnames:
        self._feature_colnames = self._encoder.get_feature_names(self._categorical_colnames).tolist()

        # Replace spaces in output columns if replace_colname_spaces is True:
        if replace_colname_spaces:
            self._feature_colnames = [s.replace(' ', '_') for s in self._feature_colnames]

        # Run transform on input (distinct) dataset:
        df_input = df.select(self._categorical_colnames).distinct().toPandas()
        df_input.reset_index(drop=True, inplace=True)
        df_one_hot = pd.DataFrame(self._encoder.transform(df_input).toarray(), columns=self._feature_colnames)

        # Create final dataframe and output:
        df_feats_pd = pd.concat([df_input, df_one_hot], axis=1)
        df_feats = self.spark.createDataFrame(df_feats_pd, schema=df_feats_pd.columns.tolist())
        # Merge results into input dataframe:
        df_result = df.join(df_feats, on=self._categorical_colnames, how='left')

        return df_result
