# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering Sample Notebook: One-Hot Encoder
# MAGIC This notebook demonstrates how to use the OneHotEncode module to create OHE features. Like all module_sample notebooks, it uses a simplified config for demonstration purposes.

# COMMAND ----------

import sys
import time
from pprint import pprint

# COMMAND ----------

# TSFA library imports:
sys.path.insert(0, '../..')
from tsfa.common.dataloader import DataLoader
from tsfa.feature_engineering.one_hot_encoder import OneHotEncode

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create test config and load data
# MAGIC
# MAGIC To load the data successfully, please ensure the **`data/dominicks_oj_data/create_oj_data_small.py` notebook is executed successfully**. The notebook will create the database and table required for this notebook.

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
        "one_hot_encoding": {
            "categorical_colnames": ["store", "brand"]
        }
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

features = config['feature_engineering']['one_hot_encoding']
pprint(features)

# COMMAND ----------

# Create featurizer object:
ohe_featurizer = OneHotEncode(spark_session=spark, categorical_colnames=features['categorical_colnames'])

# COMMAND ----------

# Run fit:
t0 = time.time()
ohe_featurizer.fit(df_train)
print(f"OHE featurizer FIT:\nRuntime: {time.time() - t0}")

# COMMAND ----------

# Run transform on Train:
t0 = time.time()
df_rslt = ohe_featurizer.transform(df_train)
r_count = df_rslt.count()
print(f"OHE featurizer Transform Train:\nRuntime: {time.time() - t0}\nTotal Rows: {r_count}")

# COMMAND ----------

# Run transform on Test:
t0 = time.time()
df_rslt = ohe_featurizer.transform(df_test)
r_count = df_rslt.count()
print(f"OHE featurizer Transform Test:\nRuntime: {time.time() - t0}\nTotal Rows: {r_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Examine output

# COMMAND ----------

# Check RSLT dataframe (used to merge to base dataframe):
display(df_rslt)
