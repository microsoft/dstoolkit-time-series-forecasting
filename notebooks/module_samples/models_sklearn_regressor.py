# Databricks notebook source
# MAGIC %md
# MAGIC # SKLearnRegressor Sample Notebook
# MAGIC The SKLearnRegressor class supports a variety of multivariate models, including Random Forest, Extra Trees, Linear Regression, XGBoost, among others. This notebook illustrates how we can leverage the SKLearnRegressor class to build these models. It provides sample configuration for each of the models and how to set up associated model hyperparameters, train a model and predict on a test dataset. To distinguish the SKLearn versions of certain models from their MLLib versions, we use the "Skl" prefix for SKLearn versions (e.g. `SklRandomForestRegressorModel` vs. `RandomForestRegressorModel`).
# MAGIC 
# MAGIC The SKLearnRegressor, like all other model classes (multivariate or univariate), can be used as part of the MLExperiment workflow to run walk-forward cross validation. See the **run_ml_experiment_mlflow** demo notebook for details.
# MAGIC 
# MAGIC **Please Note: SKLearn requires Pandas dataframes as inputs, rather than Spark dataframes. This means that each SKLearn model requires a conversion of the input dataframe to Pandas, which can be time consuming if not impossible for larger datasets. Therefore, it is recommended to only use SKLearn regressors for smaller experimentation tasks!**

# COMMAND ----------

import sys
import pandas as pd
import datetime as dt
import time
from pprint import pprint

# COMMAND ----------

# TSFF library imports:
sys.path.insert(0, '../..')
from tsff.data_prep.data_prep_utils import DataPrepUtils
from tsff.feature_engineering.features import FeaturesUtils
from tsff.models import *
from tsff.evaluation import *

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Load config and read in data

# COMMAND ----------

# Change this to one of the MLLibRegressor supported models:
MODEL_CLASS = 'SklXGBRegressorModel'

# Sample hyperparameters:
MODEL_HYPERPARAMETERS = {
    'SklLinearRegressionModel': {'fit_intercept': True, 'positive': False},
    'SklTweedieRegressorModel': {'power': 0, 'alpha': 1, 'fit_intercept': True, 'max_iter': 20},
    'SklExtraTreesRegressorModel': {'n_estimators': 50, 'max_depth': 10},
    'SklRandomForestRegressorModel': {'n_estimators': 50, 'max_depth': 10},
    'SklGradientBoostingRegressorModel': {'learning_rate': 0.1, 'n_estimators': 50, 'subsample': 0.8},
    'SklElasticNetModel': {'alpha': 1, 'l1_ratio': 0.5},
    'SklSGDRegressorModel': {'penalty': 'l2', 'alpha': 0.001, 'max_iter': 100},
    'SklSVRModel': {'kernel': 'poly', 'degree': 3},
    'SklBayesianRidgeModel': {'n_iter': 100, 'tol': 0.001},
    'SklKernelRidgeModel': {'alpha': 1, 'kernel': 'linear'},
    'SklXGBRegressorModel': {'learning_rate': 0.01, 'objective': 'reg:squarederror', 'n_estimators': 50, 'max_depth': 5, 'max_leaves': 0,
                             'grow_policy': 'lossguide', 'booster': 'gbtree'},
    'SklLGBMRegressorModel': {'learning_rate': 0.01, 'objective': 'regression', 'n_estimators': 50, 'max_depth': 5, 'class_weight': None}
}

# Holidays filepath:
HOLIDAYS_PATH = '/dbfs/FileStore/tables/holidays_1990_to_1993.json'

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
    "forecast_horizon": 4
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
# MAGIC #### Set up SKLearn Model
# MAGIC See above for model algorithm and hyperparameter definitions.

# COMMAND ----------

# Instantiate SKLearnRegressorModel object:
SKLearnModel = globals()[MODEL_CLASS](
    model_params=config['model_params'],
    dataset_schema=config['dataset_schema'],
)

# COMMAND ----------

# View some details of the model object:
print(SKLearnModel)

# COMMAND ----------

# Set feature colnames from feature engineering output:
SKLearnModel.set_feature_columns(feature_colnames=features_utils.feature_colnames)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model fit & predict

# COMMAND ----------

SKLearnModel.skl_model

# COMMAND ----------

# Fit:
SKLearnModel.fit(df_train_fe)

# COMMAND ----------

# Predict:
df_val_pred = SKLearnModel.predict(df_test_fe)

# COMMAND ----------

# View forecast output:
display(df_val_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Run some evaluations

# COMMAND ----------

# WMAPE, no grain columns:
evaluator = WMapeEvaluator()
wmape = evaluator.compute_metric_per_grain(df=df_val_pred, target_colname='quantity', forecast_colname='forecasts', grain_colnames=[])
wmape.collect()

# COMMAND ----------

# With grain cols specified:
wmape_by_grain = evaluator.compute_metric_per_grain(
    df=df_val_pred,
    target_colname='quantity',
    forecast_colname='forecasts',
    grain_colnames=['store', 'brand']
)

display(wmape_by_grain)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Try a different model
# MAGIC Modify the config a bit and train/predict another SKLearn model. No need to rerun feature engineering if no features are changing.

# COMMAND ----------

# Modify config for new model type:
MODEL_CLASS = 'SklExtraTreesRegressorModel'
config['model_params']['algorithm'] = MODEL_CLASS
config['model_params']['hyperparameters'] = MODEL_HYPERPARAMETERS[MODEL_CLASS]

# COMMAND ----------

# Instantiate SKLearn model object:
SKLearnModel = globals()[MODEL_CLASS](
    model_params=config['model_params'],
    dataset_schema=config['dataset_schema'],
)

# Set features to be used for model training:
SKLearnModel.set_feature_columns(feature_colnames=features_utils.feature_colnames)

# View some details of the model object:
print(SKLearnModel)

# COMMAND ----------

# Fit:
SKLearnModel.fit(df_train_fe)

# COMMAND ----------

# Predict:
df_val_pred = SKLearnModel.predict(df_test_fe)

# COMMAND ----------

# View forecast output:
display(df_val_pred)
