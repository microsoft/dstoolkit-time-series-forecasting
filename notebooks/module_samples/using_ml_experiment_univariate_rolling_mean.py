# Databricks notebook source
# MAGIC %md
# MAGIC # Using the MLExperiment class for Rolling Mean
# MAGIC In this notebook, we demonstrate how to leverage the MLExperiment class for Rolling Mean models.

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as F
import sys
import time
from pyspark.sql import DataFrame as SparkDataFrame
from typing import List, Dict, Tuple
from pprint import pprint

sys.path.insert(0, '../..')
from tsfa.data_prep.data_prep_utils import DataPrepUtils
from tsfa.models import RollingMeanModel
from tsfa.ml_experiment import MLExperiment
from tsfa.evaluation import WMapeEvaluator

# COMMAND ----------

rolling_mean_config = {
    "metadata": {
        "time_series": "orange_juice"
    },
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
    "feature_engineering": {},                  # No feature engineering required for rolling mean models
    "model_params": {
        "algorithm": "RollingMeanModel",
        "hyperparameters": {
            "window_size": 3
        },
        "model_name_prefix": "rolling_mean_model_regtest_small"
    },
    "evaluation": [
        {"metric": "WMapeEvaluator"},
        {"metric": "WMapeEvaluator", "express_as_accuracy": True}
    ],
    "forecast_horizon": 4,
    "results": {
        "db_name": "results",
        "table_name": "rolling_mean_orange_juice_small"
    }
}

# COMMAND ----------

data_prep = DataPrepUtils(spark_session=spark, config_dict=rolling_mean_config)
df = data_prep.load_data()

# COMMAND ----------

# Instantiate evaluation metric:
wmape_evaluator = WMapeEvaluator()

# Instantiate ML Experiment class:
exp = MLExperiment(spark_session=spark,
                   config=rolling_mean_config,
                   model_cls=RollingMeanModel,
                   evaluator_obj=wmape_evaluator)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Leveraging a single run of training, testing and evaluation

# COMMAND ----------

train_timeframe = {"start": "1990-06-20", "end": "1992-06-30"}
test_timeframe = {"start": "1992-07-01", "end": "1992-07-31"}

single_run_result = exp.single_train_test_eval(
    df, 
    train_timeframe=train_timeframe, 
    test_timeframe=test_timeframe,
    run_name="single_train_test_run",
    verbose=True
)

# COMMAND ----------

print(single_run_result.run_name)
display(single_run_result.result_df)
display(single_run_result.metrics)
print(single_run_result.train_timeframe, single_run_result.test_timeframe)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### `walk_forward_model_training`: Multiple runs of train, test and evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ##### When the user does not specify `time_splits`
# MAGIC 
# MAGIC As part of the experiment, `tsfa` by default will do splitting of the dataframe `df` that is loaded from the mount (or custom prepared by the user) based on `data_splitting` specifications in the configuraion file.

# COMMAND ----------

walk_forward_results = exp.walk_forward_model_training(df, time_splits=None, verbose=True)

# COMMAND ----------

print(walk_forward_results.avg_metrics)
print(walk_forward_results.std_metrics)

# COMMAND ----------

print(walk_forward_results.run_results[0].run_name)
display(walk_forward_results.run_results[0].result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### When the user specifies custom `time_splits` argument in the `walk_forward_model_training` method
# MAGIC 
# MAGIC TSFA will skip the in-built `data_splitting` functionality and leverage user provided custom time splits to do featurization and model 

# COMMAND ----------

custom_time_splits = {
    "training": {
        "split1": {'start': '1990-06-20', 'end': '1992-06-30'},
        "split2": {'start': '1990-06-20', 'end': '1992-07-31'}
    },
    "validation": {
      "split1": {'start': '1992-07-01', 'end': '1992-07-23'},
      "split2": {'start': '1992-08-01', 'end': '1992-08-31'}
    }
}

# COMMAND ----------

walk_forward_custom_time_splits_results = exp.walk_forward_model_training(df, time_splits=custom_time_splits, verbose=True)

# COMMAND ----------

print(walk_forward_custom_time_splits_results.avg_metrics)
print(walk_forward_custom_time_splits_results.std_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Making evaluation optional

# COMMAND ----------

exp.set_evaluator(None)
print(exp.evaluator)

# COMMAND ----------

walk_forward_no_eval_results = exp.walk_forward_model_training(df, time_splits=custom_time_splits, verbose=False)

# COMMAND ----------

print(walk_forward_no_eval_results.avg_metrics)
print(walk_forward_no_eval_results.std_metrics)
