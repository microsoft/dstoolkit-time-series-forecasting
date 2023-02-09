"""This module contains a function to save the forecast values to a table for further analysis"""

from pyspark.sql.types import StringType, DoubleType
from typing import Dict
from pyspark.sql import functions as sf
from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
from tsff.common.dataloader import DataLoader


def save_forecast_values(spark: SparkSession, df_results: SparkDataFrame, config: Dict):
    """
    Save forecast values to specified database and table in config.
    This function does require "save_forecasts.db_name", "save_forecasts.table_name",
    "feature_engineering.lags.grain_colnames", "time_colname", "target_colname" and
    "forecast_colname" to be defined in the config dictionary.

    Args:
        spark (SparkSession): Current spark session object.
        df_results (SparkDataFrame): The forecast output to save, with 'grain'
                                     (e.g. business_unit_cdv and REGION), 'time',
                                     'target', forecast_at and forecast columns
        config (Dict): The Json config file that contains arguments required.
    """
    # Get column names specified in config file
    grain_cols = config["dataset_schema"]["grain_colnames"]
    time_colname = config["dataset_schema"]["time_colname"]
    target_colname = config["dataset_schema"]["target_colname"]
    forecast_colname = config["dataset_schema"]["forecast_colname"]

    # Cast columns into required datatypes
    df_save_results = (
        df_results.select(*(sf.col(column).cast(StringType()).alias(column) for column in grain_cols),
                          sf.date_format(sf.col("forecast_at"), "yyyy-MM-dd").cast(StringType()).alias("forecast_at"),
                          sf.date_format(sf.col(time_colname), "yyyy-MM-dd").cast(StringType()).alias("forecast_for"),
                          sf.col(target_colname).cast(DoubleType()).alias("actuals"),
                          sf.col(forecast_colname).cast(DoubleType())
                          )
    )
    # Use DataLoader method to save the results to specified database and table
    db_name = config["results"]["db_name"]
    tb_name = config["results"]["table_name"]
    loader = DataLoader(spark)
    loader.save_df_to_mount(df_save_results, db_name, tb_name)
