# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering Sample Notebook: Lags
# MAGIC This notebook demonstrates how to use the LagsFeaturizer module to create lag features. Like all module_sample notebooks, it uses a simplified config for demonstration purposes.

# COMMAND ----------

import sys
import pandas as pd
import datetime as dt
import time
from pprint import pprint

# COMMAND ----------

# TSFA library imports:
sys.path.insert(0, '../..')
from tsfa.common.dataloader import DataLoader
from tsfa.feature_engineering.lags import LagsFeaturizer

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create test config and load data

# COMMAND ----------

# Create basic dummy config:
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
    'feature_engineering': {
        'feature_horizon': 4,
        "lags": {
            "colnames_to_lag": ["quantity"],
            "lag_time_steps": [1, 2, 3, 4]
        }
    }
}

# COMMAND ----------

# Load data:
loader = DataLoader(spark)
df = loader.read_df_from_mount(db_name=config['dataset']['db_name'], 
                               table_name=config['dataset']['table_name'], 
                               columns=config['dataset_schema']['required_columns'], 
                               as_pandas = False)

# COMMAND ----------

# Separate df_train and df_test:
df_train = df.filter((df.date <= "1992-06-30"))
df_test = df.filter((df.date >= "1992-07-01"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run featurizer

# COMMAND ----------

feature_params = config['feature_engineering']
lags_params = feature_params['lags']
dataset_schema = config['dataset_schema']
pprint(feature_params['feature_horizon'])
pprint(lags_params)
pprint(dataset_schema)

# COMMAND ----------

# Create featurizer object:
lags_featurizer = LagsFeaturizer(
    grain_colnames = dataset_schema['grain_colnames'],
    time_colname = dataset_schema['time_colname'],
    colnames_to_lag = lags_params['colnames_to_lag'],
    lag_time_steps = lags_params['lag_time_steps'],
    horizon = feature_params['feature_horizon'],
    ts_freq = dataset_schema['ts_freq']
)

# COMMAND ----------

# Run fit:
t0 = time.time()
lags_featurizer.fit(df_train)
r_count = lags_featurizer._padding_df.count()
print(f"Lags featurizer FIT:\nGrain: {str(dataset_schema['grain_colnames'])}\nRuntime: {time.time() - t0}\nTotal Rows: {r_count}")

# COMMAND ----------

# Run transform on train:
t0 = time.time()
df_train_trnsf = lags_featurizer.transform(df_train)
r_count = df_train_trnsf.count()
print(f"Lags featurizer Transform Train:\nGrain: {str(dataset_schema['grain_colnames'])}\nRuntime: {time.time() - t0}\nTotal Rows: {r_count}")

# COMMAND ----------

# Run transform on test:
t0 = time.time()
df_test_trnsf = lags_featurizer.transform(df_test)
r_count = df_test_trnsf.count()
print(f"Lags featurizer Transform Test:\nGrain: {str(dataset_schema['grain_colnames'])}\nRuntime: {time.time() - t0}\nTotal Rows: {r_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Examine output

# COMMAND ----------

display(lags_featurizer._padding_df)

# COMMAND ----------

# Check padding:
test = lags_featurizer._padding_df
test2 = test.filter((test.store  == '101') & (test.brand == 'minute_maid'))
display(test2)

# COMMAND ----------

# Check train:
test = df_train_trnsf
test2 = test.filter((test.store  == '101') & (test.brand == 'minute_maid'))
display(test2)

# COMMAND ----------

# Check test:
test = df_test_trnsf
test2 = test.filter((test.store  == '101') & (test.brand == 'minute_maid'))
display(test2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test padding datacheck runtime

# COMMAND ----------

t0 = time.time()
lags_featurizer._verify_padding_df(df = df_test, padding_df = lags_featurizer._padding_df)
print("Padding DF validation runtime:", time.time() - t0)
