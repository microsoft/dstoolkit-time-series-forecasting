# Databricks notebook source
# MAGIC %md
# MAGIC ## Using MLLibRegressor
# MAGIC The MLLibRegressor class supports a variety of multivariate models, including Random Forest, GBT, Linear Regression, among others. This notebook illustrates how we can leverage the MLLibRegressor class to build these models. It provides sample configuration for each of the models and how to set up associated model hyperparameters, train a model and predict on a test dataset.
# MAGIC 
# MAGIC The MLLibRegressor, like all other model classes, can be used as part of the MLExperiment workflow to run walk-forward cross validation. See the **run_ml_experiment_mlflow** demo notebook for details.

# COMMAND ----------

import sys
import time
from pprint import pprint

# COMMAND ----------

# TSFA library imports:
sys.path.insert(0, '../..')
from tsfa.data_prep.data_prep_utils import DataPrepUtils
from tsfa.feature_engineering.features import FeaturesUtils
from tsfa.models import *
from tsfa.evaluation import *

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Load config and read in data

# COMMAND ----------

# Change this to one of the MLLibRegressor supported models:
MODEL_CLASS = 'RandomForestRegressorModel'

# Sample hyperparameters:
MODEL_HYPERPARAMETERS = {
    'RandomForestRegressorModel': {'maxBins': 145, 'numTrees': 20, 'impurity': 'variance', 'maxDepth': 5, 'featureSubsetStrategy': 'auto'},
    'GBTRegressorModel': {'maxBins': 145, 'maxDepth': 5, 'maxIter': 20, 'featureSubsetStrategy': 'auto'},
    'DecisionTreeRegressorModel': {'maxBins': 145, 'maxDepth': 10},
    'LinearRegressionModel': {'maxIter': 50},
    'GeneralizedLinearRegressionModel': {'family': 'gaussian', 'link': 'identity'},
    'FMRegressorModel': {'factorSize': 4},
    'IsotonicRegressionModel': {}
}

# Holidays filepath:
HOLIDAYS_PATH = '/dbfs/FileStore/tables/holidays_1990_to_1993.json'

# COMMAND ----------

config = {
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
            "num_splits": 4,
            "rolling_origin_step_size": 2
        }
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
                "holidays_json_path": HOLIDAYS_PATH
            }
        },
        "additional_feature_colnames": ["on_promotion"]
    },
    "model_params": {
        "algorithm": MODEL_CLASS,
        "hyperparameters": MODEL_HYPERPARAMETERS[MODEL_CLASS],
        "model_name_prefix": "MLLibRegressorModel_Test"
    },
    "evaluation": [
        {"metric": "WMapeEvaluator"},
        {"metric": "WMapeEvaluator", "express_as_accuracy": 1}
    ],
    "forecast_horizon": 4,
    "results": {
        "db_name": "results",
        "table_name": "rf_orange_juice_small"
    }
}

# COMMAND ----------

# Read in data:
data_prep = DataPrepUtils(spark_session=spark, config_dict=config)
df = data_prep.load_data()
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
time_colname = config['dataset_schema']['time_colname']
print('Total time series:', df.select(*grains).distinct().count())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run feature engineering

# COMMAND ----------

# Initializing feature engineering orchestrator:
features_utils = FeaturesUtils(spark, config)
print(features_utils)

# COMMAND ----------

# Fit features on train:
t0 = time.time()
features_utils.fit(df_train)
print('Fit runtime:', time.time() - t0)
pprint(features_utils._feature_to_module_map)

# COMMAND ----------

# Transform on Train:
t0 = time.time()
df_train_fe = features_utils.transform(df_train)
print('Train FE shape:', df_train_fe.count(), '|', len(df_train_fe.columns))

# COMMAND ----------

# Transform on Test:
t0 = time.time()
df_test_fe = features_utils.transform(df_test)
print('Test FE shape:', df_test_fe.count(), '|', len(df_test_fe.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set up MLLib Model
# MAGIC See above for model algorithm and hyperparameter definitions.

# COMMAND ----------

# Instantiate MLLibRegressorModel object:
MLLibModel = globals()[MODEL_CLASS](
    model_params=config['model_params'],
    dataset_schema=config['dataset_schema'],
)

# COMMAND ----------

# View some details of the model object:
print(MLLibModel)

# COMMAND ----------

# Set feature colnames from feature engineering output:
MLLibModel.set_feature_columns(feature_colnames=features_utils.feature_colnames)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model fit & predict

# COMMAND ----------

MLLibModel.mlreg_model

# COMMAND ----------

# Fit:
MLLibModel.fit(df_train_fe)

# COMMAND ----------

# Predict:
df_test_pred = MLLibModel.predict(df_test_fe)

# COMMAND ----------

# View forecast output:
sortby_colnames = grains + [time_colname]
display(df_test_pred.sort(sortby_colnames))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run some evaluations

# COMMAND ----------

# WMAPE, no grain columns:
evaluator = WMapeEvaluator()
wmape = evaluator.compute_metric_per_grain(
    df=df_test_pred,
    target_colname=config['dataset_schema']['target_colname'],
    forecast_colname=config['dataset_schema']['forecast_colname'],
    grain_colnames=[],
)
wmape.collect()

# COMMAND ----------

# With grain cols specified:
wmape_by_grain = evaluator.compute_metric_per_grain(
    df=df_test_pred,
    target_colname=config['dataset_schema']['target_colname'],
    forecast_colname=config['dataset_schema']['forecast_colname'],
    grain_colnames=grains
)

display(wmape_by_grain)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Forecasting into the future

# COMMAND ----------

# Create a future timeframe
train_end_date = "1992-09-30"
# Recreating train data
df_new_train = data_prep.slice_data_by_time(df, start_time=None, end_time=train_end_date)

# Prepare data for the future timeframe
future_start_date = "1992-10-01"
df_future = data_prep.make_future_dataframe(df, start_time=future_start_date)
display(df_future.sort(sortby_colnames))

# COMMAND ----------

# Run features
features_utils.fit(df_new_train)
df_future_fe = features_utils.transform(df_future)
display(df_future_fe.sort(sortby_colnames))

# COMMAND ----------

# Fit MLLib model on train and predict on future
df_future_pred = MLLibModel.predict(df_future_fe)
display(df_future_pred.sort(sortby_colnames))
