# Databricks notebook source
# MAGIC %md
# MAGIC # Using the MLExperiment class for Prophet models
# MAGIC In this notebook, we demonstrate how to leverage the MLExperiment class for Prophet models.

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as sf
import sys
import time
from pyspark.sql import DataFrame as SparkDataFrame
from typing import List, Dict, Tuple
from pprint import pprint

sys.path.insert(0, '../..')
from tsff.data_prep.data_prep_utils import DataPrepUtils
from tsff.models import ProphetModel
from tsff.ml_experiment import MLExperiment
from tsff.evaluation import WMapeEvaluator

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define config and read in data

# COMMAND ----------

# MAGIC %md
# MAGIC #### Special note when working with Prophet models:
# MAGIC 
# MAGIC The following is a sample config for ProphetModel. There are 2 approaches for incorporating holidays into Prophet models:
# MAGIC 1. Add a holidays DF as a hyperparameter: This approach creates a holidays dataframe (in a specific pandas DF format, see `/dbfs/FileStore/tables/holidays_1990_to_1993.json`) and adds it as a hyperparameter during training.
# MAGIC 2. Manually create days-to-holiday features and add them to the model as regressors: This approach would create separate days-to-holiday columns during feature engineering, and adds them to the list model training variables.
# MAGIC 
# MAGIC The sample config below shows the hyperparameters approach (approach 1). However, both approaches are possible. **Please choose only one approach when running your experiment** (e.g. do not include holidays under feature_engineering and also under hyperparameters; choose one). Also note that during preliminary testing, the add-as-regressors approach runs much faster than the hyperparameters approach, as well as resulted in different forecasted values. These tradeoffs should be considered when selecting the approach.
# MAGIC 
# MAGIC Also, note that we are manually creating a feature - called 'covid' - to be added as a regressor for the model training/forecasting.

# COMMAND ----------

prophet_config = {
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
    "feature_engineering": {
        "feature_horizon": 4,
        "operations": {
            "holidays": {
                "holidays_json_path": "/dbfs/FileStore/tables/holidays_1990_to_1993.json"
            }
        },
        "additional_feature_colnames": ["on_promotion"]
    },
    "model_params": {
        "algorithm": "ProphetModel",
        "hyperparameters": {
            "interval_width": 0.95,
            "growth": "linear",
            "daily_seasonality": False,
            "weekly_seasonality": False,
            "yearly_seasonality": True,
            "seasonality_mode": "additive"
        },
        "model_name_prefix": "prophet_model_regtest_small"
    },
    "evaluation": [
        {"metric": "WMapeEvaluator"},
        {"metric": "WMapeEvaluator", "express_as_accuracy": True}
    ],
    "forecast_horizon": 4,
    "results": {
        "db_name": "results",
        "table_name": "prophet_orange_juice_small"
    }
}

# COMMAND ----------

# Read in data:
data_prep = DataPrepUtils(spark_session=spark, config_dict=prophet_config)
df = data_prep.load_data()

# COMMAND ----------

# Instantiate evaluation metric:
wmape_evaluator = WMapeEvaluator()

# Instantiate ML Experiment class:
exp = MLExperiment(spark_session=spark,
                   config=prophet_config,
                   model_cls=ProphetModel,
                   evaluator_obj=wmape_evaluator)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Single iteration of training, testing and evaluation
# MAGIC Users can leverage the `single_train_test_eval()` method to run a train-test-evaluation on a single time split. This can be viewed as a single "iteration" of walk-forward CV.

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
display(single_run_result.metrics)
print(single_run_result.train_timeframe, single_run_result.test_timeframe)

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
