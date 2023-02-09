# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering Sample Notebook: Time-Based Features
# MAGIC This notebook demonstrates how to use the BasicTimeFeatures module to create time-based features. Like all module_sample notebooks, it uses a simplified config for demonstration purposes.

# COMMAND ----------

import sys
import time
from pprint import pprint

# COMMAND ----------

import sys
sys.path.insert(0, '../..')
from tsff.common.dataloader import DataLoader
from tsff.feature_engineering.time_based import BasicTimeFeatures

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
        "basic_time_based": {
            "feature_names": ["week_of_year", "month_of_year", "week_of_month"]
        },
    }
}

# COMMAND ----------

# Load data:
loader = DataLoader(spark)
df = loader.read_df_from_mount(config['dataset']['db_name'], config['dataset']['table_name'], config['dataset_schema']['required_columns'], as_pandas = False)

# COMMAND ----------

# Separate df_train and df_test:
df_train = df.filter((df.date <= "1992-06-30"))
df_test = df.filter((df.date >= "1992-07-01"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run featurizer

# COMMAND ----------

features = config['feature_engineering']['basic_time_based']
pprint(features)

# COMMAND ----------

# Create featurizer object:
time_based_featurizer = BasicTimeFeatures(
    spark_session=spark,
    feature_names=features['feature_names'],
    time_colname=config['dataset_schema']['time_colname']
)

# COMMAND ----------

# Run transform on train:
t0 = time.time()
df_train_trnsf = time_based_featurizer.transform(df_train)
r_count = df_train_trnsf.count()
print(f"Time-Based featurizer Transform Train:\n\nRuntime: {time.time() - t0}\nTotal Rows: {r_count}")

# COMMAND ----------

# Run transform on test:
t0 = time.time()
df_test_trnsf = time_based_featurizer.transform(df_test)
r_count = df_test_trnsf.count()
print(f"Time-Based featurizer Transform Test:\nRuntime: {time.time() - t0}\nTotal Rows: {r_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Examine output

# COMMAND ----------

# Check train:
display(df_train_trnsf)

# COMMAND ----------

# Check test:
display(df_test_trnsf)
