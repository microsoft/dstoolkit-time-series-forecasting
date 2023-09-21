# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering Sample Notebook - Additional features
# MAGIC 
# MAGIC This notebook illustrates how to use the Feature Engineering - Additional features functionality.

# COMMAND ----------

import json
import sys
import time
import pyspark.sql.functions as sf
from pprint import pprint

sys.path.insert(0, '../..')
from tsfa.feature_engineering.features import FeaturesUtils
from tsfa.data_prep.data_prep_utils import DataPrepUtils
from tsfa.common.dataloader import DataLoader
from tsfa.models import RandomForestRegressorModel

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create config

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
    "feature_engineering": {
        "feature_horizon": 4,
        "operations": {
            "lags": {
                "colnames_to_lag": ["quantity"],
                "lag_time_steps": [1, 2, 3, 4]
            }
        },
        "additional_feature_colnames": ['post_coldwar']
    },
    "model_params": {
        "algorithm": "RandomForestRegressorModel",
        "hyperparameters": {
            "numTrees": 50,
            "impurity": "variance",
            "maxDepth": 5,
            "featureSubsetStrategy": "auto"
        }
    }
}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load data, create a "post_coldwar" feature (externally to `tsfa`) and split train and test dataframes
# MAGIC
# MAGIC To load the data successfully, please ensure the **`data/dominicks_oj_data/create_oj_data_small.py` notebook is executed successfully**. The notebook will create the database and table required for this notebook.

# COMMAND ----------

# Load data:
loader = DataLoader(spark)
df = loader.read_df_from_mount(
    db_name=config["dataset"]["db_name"],
    table_name=config["dataset"]["table_name"],
    columns=config["dataset_schema"]["required_columns"],
    as_pandas=False,
)

# COMMAND ----------

# Create a new column indicating post Cold War era:
time_colname = config['dataset_schema']['time_colname']
df = df.withColumn("post_coldwar", sf.when(df[time_colname] >= "1991-12-26", 1).otherwise(0))
display(df)

# COMMAND ----------

# Train-test split:
df_train = df[df["date"] < "1992-06-30"]
df_test = df[df["date"] >= "1992-07-01"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Features

# COMMAND ----------

# Call FeaturesUtils to perform feature engineering and add additional features:
features_utils = FeaturesUtils(spark, config)
df_train_feats = features_utils.fit_transform(df_train)
df_test_feats = features_utils.transform(df_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model

# COMMAND ----------

# Instantiate model object:
model = RandomForestRegressorModel(model_params=config['model_params'], dataset_schema=config['dataset_schema'])

# COMMAND ----------

# Sets the model's feature columns (X variables) based on FeaturesUtils output:
model.set_feature_columns(feature_colnames=features_utils.feature_colnames)

# COMMAND ----------

# Confirm that all relevant features are included for model training, including the "covid" feature that we manually created:
model.feature_colnames
