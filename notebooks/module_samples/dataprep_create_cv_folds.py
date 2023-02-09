# Databricks notebook source
# MAGIC %md
# MAGIC # Data Preparation: Creation of CV folds
# MAGIC This notebook illustrates how to create cross validation folds and save the timesplit JSON file to DBFS

# COMMAND ----------

import sys
import json
from pprint import pprint
sys.path.insert(0, '../..')
from tsff.common.config_manager import ConfigManager
from tsff.data_prep import data_validation
from tsff.data_prep.data_prep_utils import DataPrepUtils

# COMMAND ----------

config = {
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
    "data_splitting": {
        "train_validation_timeframe": {},
        "holdout_timeframe": {
            "start": "1992-04-01",
            "end": "1992-10-07"
        },
        "cross_validation": {
            "num_splits": 8,
            "rolling_origin_step_size": 2
        }
    },
    "forecast_horizon": 12
}

pprint(config)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create CV folds based on `data_splitting` parameters

# COMMAND ----------

# Initialize data prepration utils:
data_prep_utils = DataPrepUtils(spark, config)
df = data_prep_utils.load_data(as_pandas=False)

# COMMAND ----------

# Create cross validation folds:
cross_val_splits = data_prep_utils.train_val_holdout_split(df)
pprint(cross_val_splits)

# COMMAND ----------

# Check if time splits json is in the right format:
data_validation.validate_splits_json_config(cross_val_splits)

# Check for data leakage:
data_validation.check_data_leakage(cross_val_splits)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Modify config to drop holdout timeframe and recreate CV folds based on `data_splitting` parameters

# COMMAND ----------

config_manager = ConfigManager()
config_manager.set(config)
new_config = config_manager.modify(path_to_section=['data_splitting'],
                                   new_section_values={'holdout_timeframe': {}},
                                   in_place=False)
pprint(new_config)

# COMMAND ----------

# Initialize data prepration utils:
data_prep_utils = DataPrepUtils(spark, new_config)
# Create cross validation folds (no holdout)
cross_val_splits_no_holdout = data_prep_utils.train_val_holdout_split(df)
pprint(cross_val_splits_no_holdout)
