# Databricks notebook source
# MAGIC %md
# MAGIC # ProphetModel Sample Notebook
# MAGIC This notebook shows how to call the Prophet module. Like all module_sample notebooks, this notebook uses a simplified config for demonstration purposes.

# COMMAND ----------

import sys
import time
from pprint import pprint
import pyspark.sql.functions as sf

sys.path.insert(0, '../..')
from tsff.data_prep.data_prep_utils import DataPrepUtils
from tsff.feature_engineering.features import FeaturesUtils
from tsff.models import ProphetModel

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
    "forecast_horizon": 4
}

# COMMAND ----------

# Read data:
data_prep = DataPrepUtils(spark_session=spark, config_dict=prophet_config)
df = data_prep.load_data()

# COMMAND ----------

# Separate df_train and df_test:
df_train = df.filter((df.date <= "1992-06-30"))
df_test = df.filter((df.date >= "1992-07-01"))
print('Train shape:', df_train.count(), '|', len(df_train.columns))
print('Test shape:', df_test.count(), '|', len(df_test.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run feature engineering
# MAGIC For this Prophet model, there are no Feature Engineering operations such as lags or OHE. But since we still wish to bring in our manually created 'covid' feature, we still use FeatureUtils to carry this field through, since the output of FeatureUtils will contain all the features we specified in `additional_feature_colnames`, which gets added as regressors in the ProphetModel.

# COMMAND ----------

# Initializing feature engineering orchestrator:
features_utils = FeaturesUtils(spark, prophet_config)
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
# MAGIC ### Set up and run ProphetModel

# COMMAND ----------

Prophet = ProphetModel(
    dataset_schema=prophet_config['dataset_schema'],
    model_params=prophet_config['model_params'],
)

# COMMAND ----------

# Set feature colnames from feature engineering output:
print('Columns to be added to Prophet as regressors:', features_utils.feature_colnames)
Prophet.set_feature_columns(feature_colnames=features_utils.feature_colnames)

# COMMAND ----------

# Run Prophet fit_and_predict:
df_preds = Prophet.fit_and_predict(train_sdf=df_train_fe, test_sdf=df_test_fe)

# COMMAND ----------

# See forecasts output:
display(df_preds)
