# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluators Sample Notebook: save_forecast_values
# MAGIC 
# MAGIC This notebook is used to demonstrate the evaluations/save_forecast_values function.

# COMMAND ----------

import sys
sys.path.insert(0, '../..')
from tsfa.common.dataloader import DataLoader
from tsfa.evaluation.save_results import save_forecast_values
from tsfa.common.config_manager import ConfigManager
from pprint import pprint

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define config that specifies arguments to use

# COMMAND ----------

config_manager = ConfigManager(path_to_config='../configs/json/random_forest_config_small.json')
config = config_manager.get()
pprint(config)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Load dataset table as specified in config dictionary

# COMMAND ----------

loader = DataLoader(spark)
df_test = loader.read_df_from_mount(db_name=config["dataset"]["db_name"],
                                   table_name=config["dataset"]["table_name"],
                                   columns=config["dataset_schema"]["required_columns"],
                                   as_pandas=False)

# COMMAND ----------

# mock forecast_at and forecast values
df_test = df_test.withColumn("forecast_at", df_test.date-7)
df_test = df_test.withColumn("forecasts", df_test.quantity+10)
df_test.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Call function to save forecast values to specified database and table

# COMMAND ----------

save_forecast_values(spark, df_test, config)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test to ensure mock forecast results have been saved

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT * FROM results.rf_orange_juice_small

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DESCRIBE TABLE EXTENDED results.rf_orange_juice_small

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test exception when the argument not found in config

# COMMAND ----------

# MAGIC %md 
# MAGIC Define a config that has a different expected schema

# COMMAND ----------

config_test = {
    "data_set": {
        "db_name": "sample_data",
        "table_name": "orange_juice_small"
        },
    "dataset_schema": {
        "required_columns": [
            "store", 
            "brand", 
            "date", 
            "quantity"
        ]
    }
}

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **The below call to the function should fail with KeyError.**

# COMMAND ----------

save_forecast_values(spark, df_test, config_test)
