# Databricks notebook source
# MAGIC %md
# MAGIC # RollingMean Sample Notebook
# MAGIC This notebook shows how to call the Rolling Mean module. Like all module_sample notebooks, this notebook uses a simplified config for demonstration purposes.

# COMMAND ----------

import sys
import time
from pprint import pprint

sys.path.insert(0, '../..')
from tsff.data_prep.data_prep_utils import DataPrepUtils
from tsff.models import RollingMeanModel

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define config and read in data

# COMMAND ----------

rolling_mean_config = {
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
    "feature_engineering": {},      # No feature_engineering required for rolling mean model
    "model_params": {
        "algorithm": "RollingMeanModel",
        "hyperparameters": {
            'window_size': 3
        },
        "model_name_prefix": "rolling_mean_model_business_unit_cdv_bdc_size_cdv_REGION"
    },
    "forecast_horizon": 4,
}

# COMMAND ----------

# Read data:
data_prep = DataPrepUtils(spark_session=spark, config_dict=rolling_mean_config)
df = data_prep.load_data()

# COMMAND ----------

# Separate df_train and df_test:
df_train = df.filter((df.date <= "1992-06-30"))
df_test = df.filter((df.date >= "1992-07-01"))
print('Train shape:', df_train.count(), '|', len(df_train.columns))
print('Test shape:', df_test.count(), '|', len(df_test.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test model implementation

# COMMAND ----------

RMModel = RollingMeanModel(
    dataset_schema=rolling_mean_config['dataset_schema'],
    model_params=rolling_mean_config['model_params'],
)

# COMMAND ----------

df_preds = RMModel.fit_and_predict(train_sdf=df_train, test_sdf=df_test)

# COMMAND ----------

display(df_preds)
