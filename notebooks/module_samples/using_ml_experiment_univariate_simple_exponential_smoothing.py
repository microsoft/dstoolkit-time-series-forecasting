# Databricks notebook source
# MAGIC %md
# MAGIC # Using the MLExperiment class for Simple Exponential Smoothing
# MAGIC In this notebook, we demonstrate how to leverage the MLExperiment class for Simple Exponential Smoothing models.

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as F
import sys
import time
from pyspark.sql import DataFrame as SparkDataFrame
from typing import List, Dict, Tuple
from pprint import pprint

sys.path.insert(0, '../..')
from tsff.data_prep.data_prep_utils import DataPrepUtils
from tsff.models import SimpleExpSmoothingModel
from tsff.ml_experiment import MLExperiment
from tsff.evaluation import WMapeEvaluator

# COMMAND ----------

simple_exp_smoothing_config = {
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
    "feature_engineering": {},               # Feature engineering not necessary for exponential smoothing
    "model_params": {
        "algorithm": "SimpleExpSmoothingModel",
        "hyperparameters": {
            "optimized": True,
            "use_brute": True
        },
        "model_name_prefix": "exp_smoothing_model_regtest_small"
    },
    "forecast_horizon": 4,
    "results": {
        "db_name": "results",
        "table_name": "exp_smoothing_orange_juice_small"
    }
}

# COMMAND ----------

data_prep = DataPrepUtils(spark_session=spark, config_dict=simple_exp_smoothing_config)
df = data_prep.load_data()

# COMMAND ----------

# Instantiate evaluation metric:
wmape_evaluator = WMapeEvaluator()

# Instantiate ML Experiment class:
exp = MLExperiment(spark_session=spark,
                   config=simple_exp_smoothing_config,
                   model_cls=SimpleExpSmoothingModel,
                   evaluator_obj=wmape_evaluator)

# COMMAND ----------

# MAGIC %md
# MAGIC #### When the user does not specify `time_splits` argument in the `walk_forward_model_training` method
# MAGIC 
# MAGIC As part of the experiment, `tsff` by default will do splitting of the dataframe `df` that is loaded from the mount (or custom prepared by the user) based on `data_splitting` specifications in the configuraion file.

# COMMAND ----------

walk_forward_results = exp.walk_forward_model_training(df, time_splits=None, verbose=True)

# COMMAND ----------

pprint(walk_forward_results)

# COMMAND ----------

display(walk_forward_results.run_results[0].result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### When the user specifies custom `time_splits` argument in the `walk_forward_model_training` method
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

walk_forward_custom_splits_results = exp.walk_forward_model_training(df, time_splits=custom_time_splits, verbose=True)

# COMMAND ----------

pprint(walk_forward_custom_splits_results)

# COMMAND ----------

display(walk_forward_custom_splits_results.run_results[0].result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Leveraging a single run of training, testing and evaluation

# COMMAND ----------

result = exp.single_train_test_eval(df, 
                                    train_timeframe=custom_time_splits['training']['split1'], 
                                    test_timeframe=custom_time_splits['validation']['split1'],
                                    run_name="single_train_test_run",
                                    verbose=True)

# COMMAND ----------

pprint(result)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Making evaluation optional

# COMMAND ----------

exp.set_evaluator(None)
print(exp.evaluator)

# COMMAND ----------

result_no_eval = exp.single_train_test_eval(df, 
                                            train_timeframe=custom_time_splits['training']['split1'], 
                                            test_timeframe=custom_time_splits['validation']['split1'],
                                            run_name="single_train_test_run_no_eval",
                                            verbose=True)

# COMMAND ----------

pprint(result_no_eval)
