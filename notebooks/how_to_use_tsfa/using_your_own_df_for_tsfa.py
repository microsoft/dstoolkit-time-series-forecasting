# Databricks notebook sourcesource
# MAGIC %md
# MAGIC # Using your own dataframe for TSFA
# MAGIC 
# MAGIC This notebook demonstrates how you can load a dataframe that you prepared external to TSFA, make transformations to it, and then leverage TSFA with the in-memory dataframe (without needing to save to a delta table). This notebook uses a simplified config for demonstration purposes.
# MAGIC
# MAGIC To load the data successfully, please ensure the **`data/dominicks_oj_data/create_oj_data_small.py` notebook is executed successfully**. The notebook will create the database and table required for this notebook.
# MAGIC Additionally, ensure the **`data/dominicks_oj_data/holidays_1990_to_1993.json`** file is uploaded to `/dbfs/FileStore/tables/holidays_1990_to_1993.json`.

# COMMAND ----------

import json
import sys
import time
import pyspark.sql.functions as sf
from pprint import pprint

sys.path.insert(0, '../..')
from tsfa.common.config_manager import ConfigManager
from tsfa.data_prep.data_prep_utils import DataPrepUtils
from tsfa.feature_engineering.features import FeaturesUtils
from tsfa.models import RandomForestRegressorModel

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 0: Load config for experiments with orange juice dataset

# COMMAND ----------

cnf_manager = ConfigManager(path_to_config="../configs/json/random_forest_config_small.json")
config = cnf_manager.get()
pprint(config)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Load sample dataset

# COMMAND ----------

# Load the dataset
data_prep_utils = DataPrepUtils(spark_session=spark, config_dict=config)
df = data_prep_utils.load_data()
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Make data transformations, do data cleaning, create new features and prepare it for running an experiment
# MAGIC 
# MAGIC Example: Create a "post_coldwar" feature as a binary time indicator (This new column represents an **additional feature** created external to `tsff`)

# COMMAND ----------

time_colname = config['dataset_schema']['time_colname']
new_df = df.withColumn("post_coldwar", sf.when(df[time_colname] >= "1991-12-26", 1).otherwise(0))
display(new_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Modify the config to include these additional features to drive your experiment on your processed dataframe `new_df`

# COMMAND ----------

# Include the manually created "post_coldwar" field in the updated config
new_additional_features_dict = {
    "additional_feature_colnames": ['on_promotion', 'post_coldwar']
}
_ = cnf_manager.modify(path_to_section=['feature_engineering'], 
                       new_section_values=new_additional_features_dict,
                       in_place=True)

# Update required columns
new_required_columns_dict = {
    "required_columns": config['dataset_schema']["required_columns"] + ["post_coldwar"]
}
_ = cnf_manager.modify(path_to_section=['dataset_schema'], 
                       new_section_values=new_required_columns_dict,
                       in_place=True)

# View updated config
config = cnf_manager.get()
pprint(config)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Step 3: Create train / test data

# COMMAND ----------

df_train = new_df[new_df["date"] <= "1992-06-30"]
df_test = new_df[new_df["date"] >= "1992-07-01"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Create features

# COMMAND ----------

features_utils = FeaturesUtils(spark, config)
df_train_feats = features_utils.fit_transform(df_train)
df_test_feats = features_utils.transform(df_test)

# COMMAND ----------

display(df_test_feats)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5: Train a model and predict on some test data

# COMMAND ----------

model = RandomForestRegressorModel(model_params=config['model_params'], dataset_schema=config['dataset_schema'])
model.set_feature_columns(feature_colnames=features_utils.feature_colnames)

# COMMAND ----------

# Verify that the feature columns are correct:
model.feature_colnames

# COMMAND ----------

model.fit(df_train_feats)

# COMMAND ----------

df_test_pred = model.predict(df_test_feats)
display(df_test_pred)
