# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering Sample Notebook: Holidays
# MAGIC This notebook demonstrates how to use the DaysToHolidays module to create holiday features. Like all module_sample notebooks, it uses a simplified config for demonstration purposes.

# COMMAND ----------

import sys
import json
from pprint import pprint
import time

import sys
sys.path.insert(0, '../..')
from tsfa.feature_engineering.holidays import DaysToHolidays
from tsfa.data_prep.data_prep_utils import DataPrepUtils
from tsfa.common.dataloader import DataLoader

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read config and load dataset

# COMMAND ----------

config_dict = {
    "dataset": {
        "db_name": "sample_data",
        "table_name": "orange_juice_small"
    },
    "dataset_schema": {
        "required_columns": ["date",
                             "quantity",
                             "store",
                             "brand",
                             "on_promotion"],
        "grain_colnames": ["store", "brand"],
        "time_colname": "date",
        "target_colname": "quantity",
        "forecast_colname": "forecasts",
        "ts_freq": "W-WED"
    },
    "feature_engineering": {
        "operations": {
            "holidays": {
                "holidays_json_path": "/dbfs/FileStore/tables/holidays_1990_to_1993.json"
            }
        }
    }
}

# COMMAND ----------

# load data
loader = DataLoader(spark)
df = loader.read_df_from_mount(db_name=config_dict["dataset"]["db_name"],
                               table_name=config_dict["dataset"]["table_name"],
                               columns=config_dict["dataset_schema"]["required_columns"],
                               as_pandas=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Days to Holidays featurizer

# COMMAND ----------

holidays_features = config_dict['feature_engineering']['operations']['holidays']
pprint(holidays_features)

# COMMAND ----------

# initialize DaysToHolidays
days_to_holidays = DaysToHolidays(spark_session=spark,
                                  time_colname=config_dict["dataset_schema"]["time_colname"],
                                  holidays_path=holidays_features['holidays_json_path'])

# COMMAND ----------

# check out holidays
days_to_holidays.holidays

# COMMAND ----------

df_holidays_feats = days_to_holidays.transform(df)
display(df_holidays_feats)

# COMMAND ----------

df_holidays_feats.count()
