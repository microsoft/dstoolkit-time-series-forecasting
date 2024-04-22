# Databricks notebook source
# MAGIC %md
# MAGIC # Using the MLExperiment class with Error Analysis
# MAGIC In this notebook, we demonstrate how to leverage the MLExperiment class and present the capapilites of the Error Analysis class:
# MAGIC - Plot a distribution of the error for key columns
# MAGIC - Plot the error as a function of the time
# MAGIC - Plot the error as a cohort of the time and the walk number
# MAGIC - On the experiment side, we show how to:
# MAGIC - Create and bring in your own additional features.
# MAGIC - Set up MLExperiment with your choice of model and evaluator (RandomForest and WMAPE in this example).
# MAGIC - Leverage the `single_train_test_eval` method to run a "single fold" of model training-testing-evaluation.
# MAGIC - Leverage the `walk_forward_model_training` method to run a full walk-forward cross validation experiment, either with auto-generated data splits or manually provided data splits.
# MAGIC - How to use `walk_forward_model_training` to forecast into the future.
# MAGIC 
# MAGIC Note: To see an ends-to-end workflow where the MLExperiment class is used to run a walk-forward cross validation ML experiment and log its results to MLFlow, refer to the **run_ml_experiment_mlflow** notebook.
# MAGIC
# MAGIC To load the data successfully, please ensure the **`data/dominicks_oj_data/create_oj_data_small.py` notebook is executed successfully**. The notebook will create the database and table required for this notebook.
# MAGIC Additionally, ensure the **`data/dominicks_oj_data/holidays_1990_to_1993.json`** file is uploaded to `/dbfs/FileStore/tables/holidays_1990_to_1993.json`.

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as sf
import time
from pyspark.sql import DataFrame as SparkDataFrame
from typing import List, Dict, Tuple
from pprint import pprint

import sys
sys.path.insert(0, '../..')
from tsfa.common.config_manager import ConfigManager
from tsfa.data_prep.data_prep_utils import DataPrepUtils
from tsfa.models import RandomForestRegressorModel
from tsfa.ml_experiment import MLExperiment
from tsfa.evaluation import WMapeEvaluator
from tsfa.error_analysis import ErrorAnalysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define config and read in data

# COMMAND ----------

# Configuration file to use:
config_filename = "random_forest_config_small.json" # "YOUR_CONFIG_JSON_FILE"
config_path = f"../configs/json/{config_filename}"

# ConfigParser contains helper methods to read in the contents a Config file. Future versions will also have
# methods to validate the contents of the config.
cnf_manager = ConfigManager(path_to_config=config_path)
config = cnf_manager.get()
pprint(config)

# COMMAND ----------

# Read in data:
data_prep = DataPrepUtils(spark_session=spark, config_dict=config)
df = data_prep.load_data()

# COMMAND ----------

# Instantiate evaluation metric:
wmape_evaluator = WMapeEvaluator()

# Instantiate ML Experiment class:
exp = MLExperiment(spark_session=spark,
                   config=config,
                   model_cls=RandomForestRegressorModel,
                   evaluator_obj=wmape_evaluator)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Performing Walk-Forward Cross-Validation
# MAGIC For full CV runs, users can use the built-in `walk_forward_model_training()` method. This method calls the `single_train_test_eval()` method demonstrated above in a multithreaded manner; data splits are constructed or defined depending on your config.

# COMMAND ----------

# MAGIC %md
# MAGIC **1. When the user does not specify custom `time_splits` argument in the `walk_forward_model_training` method**
# MAGIC
# MAGIC As part of the experiment, TSFA by default will do splitting of the dataframe `df` that is loaded from the mount (or custom prepared by the user) based on `data_splitting` specifications in the configuraion file.

# COMMAND ----------

walk_forward_results = exp.walk_forward_model_training(df, time_splits=None, verbose=True)

# COMMAND ----------

print(walk_forward_results.avg_metrics)
print(walk_forward_results.std_metrics)

# COMMAND ----------

print(walk_forward_results.run_results[0].run_name)
display(walk_forward_results.run_results[0].result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Building the dataframe for the error analysis throw multi walk forwards.
# MAGIC caching the results in pandas dataframe and then converting back to spark dataframe.
# MAGIC this is done to avoid the long run time of the error analysis.
# MAGIC in this section the keys, prediction column, target column and the time column are fixed.

# COMMAND ----------

all_results_ = pd.DataFrame()
for i, result in enumerate(walk_forward_results.run_results):
    print(f'Walk number {len(walk_forward_results.run_results) - i}')
    df_result = result.result_df.toPandas()
    df_result['walk'] = len(walk_forward_results.run_results) - i
    all_results_ = pd.concat([all_results_, df_result])


all_results = spark.createDataFrame(all_results_)
result_df_ = all_results.groupby(['store', 'brand','date']).mean()
result_df_ = result_df_.withColumnRenamed("avg(forecasts)","forecasts")
result_df_ = result_df_.withColumnRenamed("avg(quantity)", "quantity")
result_df = spark.createDataFrame(result_df_.toPandas())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Activating error anaylsis functions.

# COMMAND ----------

myErrorAnalysis = ErrorAnalysis(config=config)

# COMMAND ----------

# MAGIC %md
# MAGIC ### plotting the error as a cohort plot.
# MAGIC The cohort plot is a plot of the error as a function of the time and the walk number.
# MAGIC the vmin and vmax are for the colorbar.
# MAGIC In this plot you can see the error as a function of the time and the walk number.
# MAGIC You can see the involvement of the model over time and walk number.

# COMMAND ----------

myErrorAnalysis.cohort_plot(all_results=all_results, walk_name = 'walk', vmin = 0.1, vmax = 1.5)

# COMMAND ----------
# MAGIC %md

# MAGIC ### Using the plot hist function to show different distributions.
# MAGIC The plot hist function plots the error distribution of the key columns
# MAGIC Why do we need to plot for different keys combinations?
# MAGIC When we look only on the store level or the brand level we understand better how to model preforme.
# MAGIC But we miss some cruicel information, for example, if we have a store that is doing well in the model and we have a brand that is doing well in the model, but when we combine them together we get a bad model.
# MAGIC Some stores and brands would need different approches, and with the plot hist function we can see it.

# COMMAND ----------

myErrorAnalysis.plot_hist(df=result_df,keys=['store', 'brand'], bins=20, precentile=0.95, cut=False)
# We have a distribution of the error for store, brand combination

# COMMAND ----------

myErrorAnalysis.plot_hist(df=result_df,keys=['store'], bins=20, precentile=0.95, cut=False)
# We have a distribution of the error for store
# COMMAND ----------

myErrorAnalysis.plot_hist(df=result_df,keys=['brand'], bins=20, precentile=0.95, cut=False)
# We have a distribution of the error for brand 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using the plot time function to show the error as a function of the time
# MAGIC Overall the model could perform well, but in some time periods it could perform badly.
# MAGIC Covid, holidays, seasonal events could affect the model performance.
# MAGIC With this function we can get clarity on the time periods that the model perform well or badly.

# COMMAND ----------

myErrorAnalysis.plot_time(df=result_df)
# 
# COMMAND ----------

# MAGIC %md
# MAGIC ### Using the plot examples function to show examples of the error for each key combination
# MAGIC The plot examples function is calculating the best and the wrost preforming examples for each key combination.
# MAGIC And it plots the results over time
# MAGIC This function is very useful to understand the model boundaries of best and wrost performing examples.
# MAGIC The function brings more clarity to why and when the model performance well or badly.

# COMMAND ----------

myErrorAnalysis.plot_examples(df=result_df,
                                top=True,
                                num_of_pairs=5
                                )

# COMMAND ----------

myErrorAnalysis.plot_examples(df=result_df,
                                top=False,
                                num_of_pairs=5
                                )


# COMMAND ----------
