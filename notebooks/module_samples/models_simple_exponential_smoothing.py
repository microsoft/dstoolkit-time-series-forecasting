# Databricks notebook source
# MAGIC %md
# MAGIC # SimpleExpSmoothingModel Sample Notebook
# MAGIC This notebook shows how to call the Simple Exponential Smoothing module. Like all module_sample notebooks, this notebook uses a simplified config for demonstration purposes.

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as F
import sys
import time
from pyspark.sql import DataFrame as SparkDataFrame
from typing import List, Dict, Tuple

sys.path.insert(0, '../..')
from tsfa.data_prep.data_prep_utils import DataPrepUtils
from tsfa.feature_engineering.features import FeaturesUtils
from tsfa.models import SimpleExpSmoothingModel

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create config and read in data

# COMMAND ----------

simple_exp_smoothing_config = {
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
    "feature_engineering": {},
    "model_params": {
        "algorithm": "SimpleExpSmoothingModel",
        "hyperparameters": {
            'initialization_method': None,
            'initial_level': None,
            'smoothing_level': None,
            'optimized': True,
            'start_params': None,
            'use_brute': True
        },
        "model_name_prefix": "ema_model_business_unit_cdv_bdc_size_cdv_REGION"
    },
    "forecast_horizon": 4
}

# COMMAND ----------

# Read data:
data_prep = DataPrepUtils(spark_session=spark, config_dict=simple_exp_smoothing_config)
df = data_prep.load_data()

# COMMAND ----------

# Separate df_train and df_test:
df_train = df.filter((df.date <= "1992-06-30"))
df_test = df.filter((df.date >= "1992-07-01"))
print('Train shape:', df_train.count(), '|', len(df_train.columns))
print('Test shape:', df_test.count(), '|', len(df_test.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Model Implementation

# COMMAND ----------

Simple_Exp_Smoothing_model = SimpleExpSmoothingModel(
    dataset_schema=simple_exp_smoothing_config['dataset_schema'],
    model_params=simple_exp_smoothing_config['model_params'],
)

# COMMAND ----------

df_preds = Simple_Exp_Smoothing_model.fit_and_predict(train_sdf=df_train, test_sdf=df_test)

# COMMAND ----------

display(df_preds)

# COMMAND ----------

df_preds.count()
