# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering Sample Notebook: Complete Feature Engineering Run
# MAGIC This notebook demonstrates how to use the FeaturesUtils module to run Feature Engineering operations. Like all module_sample notebooks, it uses a simplified config for demonstration purposes.
# MAGIC
# MAGIC To load the data successfully, please ensure the **`data/dominicks_oj_data/create_oj_data_small.py` notebook is executed successfully**. The notebook will create the database and table required for this notebook.
# MAGIC Additionally, ensure the **`data/dominicks_oj_data/holidays_1990_to_1993.json`** file is updated to `/dbfs/FileStore/tables/holidays_1990_to_1993.json`.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setup

# COMMAND ----------

import sys
import pandas as pd
import datetime as dt
import time
from pprint import pprint

# COMMAND ----------

# TSFA library imports:
sys.path.insert(0, '../..')
from tsfa.data_prep.data_prep_utils import DataPrepUtils
from tsfa.feature_engineering.features import FeaturesUtils

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load config and read in data

# COMMAND ----------

# Define config:
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
    "feature_engineering": {
        "feature_horizon": 4,
        "operations": {
            "one_hot_encoding": {
                "categorical_colnames": ["store", "brand"]
            },
            "basic_time_based": {
                "feature_names": ["week_of_year", "month_of_year", "week_of_month"]
            },
            "lags": {
                "colnames_to_lag": ["quantity"],
                "lag_time_steps": [1, 2, 3, 4]
            },
            "holidays": {
                "holidays_json_path": "/dbfs/FileStore/tables/holidays_1990_to_1993.json"
            }
        },
        "additional_feature_colnames": ["on_promotion"]
    },
    "forecast_horizon": 4
}

# COMMAND ----------

# Read in data:
data_prep_utils = DataPrepUtils(spark, config)
df = data_prep_utils.load_data(as_pandas=False)
print(df.count(), '|', len(df.columns))

# COMMAND ----------

# Separate df_train and df_test:
df_train = df.filter((df.date <= "1992-06-30"))
df_test = df.filter((df.date >= "1992-07-01"))
print('Train shape:', df_train.count(), '|', len(df_train.columns))
print('Test shape:', df_test.count(), '|', len(df_test.columns))

# COMMAND ----------

# How many time series:
grains = config['dataset_schema']['grain_colnames']
print('Total time series:', df.select(*grains).distinct().count())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run feature engineering orchestrator

# COMMAND ----------

# Initializing feature engineering orchestrator:
features_utils = FeaturesUtils(spark, config)
print(features_utils)

# COMMAND ----------

# Fit on train:
t0 = time.time()
features_utils.fit(df_train)
print('Fit runtime:', time.time() - t0)
pprint(features_utils._feature_to_module_map)

# COMMAND ----------

# Transform on Train:
t0 = time.time()
df_train_fe = features_utils.transform(df_train)
print('Train FE shape:', df_train_fe.count(), '|', len(df_train_fe.columns))
print('Transform Train runtime:', time.time() - t0)

# COMMAND ----------

# Transform on Test:
t0 = time.time()
df_test_fe = features_utils.transform(df_test)
print('Test FE shape:', df_test_fe.count(), '|', len(df_test_fe.columns))
print('Transform Test runtime:', time.time() - t0)

# COMMAND ----------

# MAGIC %md
# MAGIC #### View the results

# COMMAND ----------

# View train:
display(df_train_fe)

# COMMAND ----------

# View test:
display(df_test_fe)

# COMMAND ----------

# Final columns:
df_train_fe.columns
