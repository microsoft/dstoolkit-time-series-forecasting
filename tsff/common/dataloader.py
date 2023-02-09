"""
This module contains the DataLoader class, which is used to read and save data
into and out of spark / pandas dataframes.
"""

import pandas as pd
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from typing import Union, List


class DataLoader(object):
    """Class that will read, save and merge datasets."""

    def __init__(self, spark_session: SparkSession):
        """
        Args:
            spark_session (SparkSession): Existing spark session.
        """
        self.spark = spark_session

    def read_df_from_mount(
        self,
        db_name: str,
        table_name: str,
        columns: List[str] = None,
        as_pandas: bool = True
    ) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Reads a given table from mount into a Spark or pandas dataframe.

        Args:
            db_name (str): The name of the database.
            table_name (str): The name of the database table.
            columns (List[str]): List of columns to load when reading the dataframe.
            as_pandas (bool): True/False indicator for returning as a pandas dataframe.

        Returns:
            df (SparkDataframe or pd.Dataframe): Dataframe the database table is read to.

        Raises:
            ValueError: When table name is not specified.
        """
        if (not table_name) or (not db_name):
            raise ValueError("Provide a database name and a table name to load!")

        if columns:
            columns_str = ", ".join(columns)
        else:
            columns_str = "*"

        query = f"select {columns_str} from {db_name}.{table_name}"
        df = self.spark.sql(query)

        # Return pandas dataframe if as_pandas is True
        return df.toPandas() if as_pandas else df

    def save_df_to_mount(self, df: Union[pd.DataFrame, SparkDataFrame], db_name: str, table_name: str):
        """
        Save Pandas or Spark dataframe to a table.

        Args:
            df (SparkDataframe or pd.Dataframe): Dataframe that is being saved to a table.
            db_name (str): The name of the database.
            table_name (str): The name of the database table.

        Raises:
            ValueError: When db_name name is not specified.
                        When table_name is not specified.
            TypeError: When input dataframe is not SparkDataFrame or pd.DataFrame.
        """
        if isinstance(df, pd.DataFrame) and not df.empty:
            spark_df = self.spark.createDataFrame(df, schema=df.columns.tolist())
        elif isinstance(df, SparkDataFrame) and not df.rdd.isEmpty():
            spark_df = df
        else:
            raise TypeError("Input dataframe should be pandas or spark type")

        if not db_name:
            raise ValueError("Database name not specified!")

        if not table_name:
            raise ValueError("Table name not specified!")

        (
            spark_df.write.mode("overwrite")
            .option("overwriteSchema", "true")
            .format("delta")
            .saveAsTable(f"{db_name}.{table_name}")
        )
