# Databricks notebook source
# MAGIC %md
# MAGIC # Using the MLExperiment class
# MAGIC In this notebook, we demonstrate how to leverage the MLExperiment class and some of the flexibility it offers. In particular, this notebook shows how you can:
# MAGIC - Create and bring in your own additional features.
# MAGIC - Set up MLExperiment with your choice of model and evaluator (RandomForest and WMAPE in this example).
# MAGIC - Leverage the `single_train_test_eval` method to run a "single fold" of model training-testing-evaluation.
# MAGIC - Leverage the `walk_forward_model_training` method to run a full walk-forward cross validation experiment, either with auto-generated data splits or manually provided data splits.
# MAGIC - How to use `walk_forward_model_training` to forecast into the future.
# MAGIC 
# MAGIC Note: To see an end-to-end workflow where the MLExperiment class is used to run a walk-forward cross validation ML experiment and log its results to MLFlow, refer to the **run_ml_experiment_mlflow** notebook.

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as sf
import time
from pyspark.sql import DataFrame as SparkDataFrame
from typing import List, Dict, Tuple
from pprint import pprint

import sys
sys.path.insert(0, '../..')
from tsff.common.config_manager import ConfigManager
from tsff.data_prep.data_prep_utils import DataPrepUtils
from tsff.models import RandomForestRegressorModel
from tsff.ml_experiment import MLExperiment
from tsff.evaluation import WMapeEvaluator

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define config and read in data

# COMMAND ----------

# Configuration file to use:
config_filename = "random_forest_config_small.json" # "YOUR_CONFIG_JSON_FILE"
config_path = f"../configs/json/{config_filename}"

# ConfigParser contains helper methods to read in the contents a Config file. Future versions will also have
# methods to validate the contents of the config.
cnf_manager = ConfigManager(path_to_config=config_path)
config = cnf_manager.get()
pprint(config)

# COMMAND ----------

# Read in data:
data_prep = DataPrepUtils(spark_session=spark, config_dict=config)
df = data_prep.load_data()

# COMMAND ----------

# Instantiate evaluation metric:
wmape_evaluator = WMapeEvaluator()

# Instantiate ML Experiment class:
exp = MLExperiment(spark_session=spark,
                   config=config,
                   model_cls=RandomForestRegressorModel,
                   evaluator_obj=wmape_evaluator)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Single iteration of training, testing and evaluation
# MAGIC Users can leverage the `single_train_test_eval()` method to run a train-test-evaluation on a single time split. This can be viewed as a single "iteration" of walk-forward CV.

# COMMAND ----------

train_timeframe = {"start": "1990-06-20", "end": "1992-06-30"}
test_timeframe = {"start": "1992-07-01", "end": "1992-07-31"}

result = exp.single_train_test_eval(
    df,
    train_timeframe=train_timeframe,
    test_timeframe=test_timeframe,
    run_name="single_train_test_run",
    verbose=True
)

# COMMAND ----------

print(result.run_name)
display(result.metrics)
print(result.train_timeframe, result.test_timeframe)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Performing Walk-Forward Cross-Validation
# MAGIC For full CV runs, users can use the built-in `walk_forward_model_training()` method. This method calls the `single_train_test_eval()` method demonstrated above in a multithreaded manner; data splits are constructed or defined depending on your config.

# COMMAND ----------

# MAGIC %md
# MAGIC **1. When the user does not specify custom `time_splits` argument in the `walk_forward_model_training` method**
# MAGIC 
# MAGIC As part of the experiment, TSFF by default will do splitting of the dataframe `df` that is loaded from the mount (or custom prepared by the user) based on `data_splitting` specifications in the configuraion file.

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
# MAGIC **2. When the user specifies custom `time_splits` argument in the `walk_forward_model_training` method**
# MAGIC 
# MAGIC TSFF will skip the in-built `data_splitting` functionality and leverage user provided custom time splits to do featurization and model 

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
# MAGIC **3. Evaluation step is optional**
# MAGIC 
# MAGIC When evaluators are set to None, `walk_forward_model_training()` only performs training and predicting, and skips evaluations.

# COMMAND ----------

exp.set_evaluator(None)
print(exp.evaluator)

# COMMAND ----------

walk_forward_no_eval_results = exp.walk_forward_model_training(df, time_splits=custom_time_splits, verbose=True)

# COMMAND ----------

print(walk_forward_no_eval_results.avg_metrics)
print(walk_forward_no_eval_results.std_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Examples of validation checks that happens within TSFF

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1. Time splits wrong format
# MAGIC Should raise KeyError.

# COMMAND ----------

time_splits_wrong_format = {
    "training": {
        "split1": {"start": "1990-01-01", "end": "1991-12-01"}
    },
    "validation": {
      "split1": {"start": "1991-12-02"},
    }
}

# COMMAND ----------

_ = exp.walk_forward_model_training(df, time_splits=time_splits_wrong_format, verbose=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2. Time splits data leakage
# MAGIC Should raise ValueError.

# COMMAND ----------

time_splits_data_leakage = {
    "training": {
        "split1": {"start": "1990-01-01", "end": "1990-12-01"}
    },
    "validation": {
      "split1": {"start": "1990-11-02", "end": "1991-02-01"}
    }
}

# COMMAND ----------

_ = exp.walk_forward_model_training(df, time_splits=time_splits_data_leakage, verbose=True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### 3. Training data does not connect with validation data
# MAGIC 
# MAGIC Should raise an exception since we are missing some dates in the middle according to the specified frequency

# COMMAND ----------

time_splits_train_test_disconnect = {
    "training": {
        "split1": {"start": "1990-06-20", "end": "1991-12-31"}
    },
    "validation": {
        "split1": {"start": "1992-02-01", "end": "1992-03-01"}  # One month disconnected
    }
}

# COMMAND ----------

_ = exp.walk_forward_model_training(df, time_splits=time_splits_train_test_disconnect, verbose=True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Forecasting into the future

# COMMAND ----------

## Timeframe for df
time_colname = "date"
df_timestamps_pd = df.select(time_colname).dropDuplicates().toPandas()
### Assuming dates are in string format, we first parse the dates as datetimes using time_format
df_timestamps_pd[time_colname] = pd.to_datetime(df_timestamps_pd[time_colname], format="%Y-%m-%d")
### start and end dates of df timeframe
df_start, df_end = min(df_timestamps_pd[time_colname]), max(df_timestamps_pd[time_colname])
print(df_start, df_end)

# COMMAND ----------

# MAGIC %md
# MAGIC Doing walk forward model training with two splits of data (Forecast horizon: 12 weeks)
# MAGIC - Split 1: Training up to June 30, 1992; Validation set: July 1, 1992 to July 23, 1992 (Both train and validation completely within the df timeframe).
# MAGIC - Split 2: Training up to Sept 30, 1992; Validation set (future forecast): Oct 1, 1992 to Oct 31, 1992 (Validation set overlaps with the df timeframe and extends into the future).

# COMMAND ----------

future_forecast_time_splits = {
    "training": {
        "split1": {"start": "1990-06-20", "end": "1992-06-30"},
        "split2": {"start": "1990-06-20", "end": "1992-09-30"}
    },
    "validation": {
      "split1": {"start": "1992-07-01", "end": "1992-07-23"},
      "split2": {"start": "1992-10-01", "end": "1992-10-31"}
    }
}
pprint(future_forecast_time_splits)

# COMMAND ----------

walk_forward_future_forecast_results = exp.walk_forward_model_training(df, time_splits=future_forecast_time_splits, verbose=True)

# COMMAND ----------

for i in range(len(future_forecast_time_splits)):
    run_result = walk_forward_future_forecast_results.run_results[i]
    print(f"Train_timeframe ({run_result.run_name}): {run_result.train_timeframe}")
    print(f"Test_timeframe ({run_result.run_name}): {run_result.test_timeframe}")
    display(run_result.result_df)
