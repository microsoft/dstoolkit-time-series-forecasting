# Databricks notebook source
# MAGIC %md
# MAGIC # Create Configuration File - Exponential Smoothing Example
# MAGIC 
# MAGIC This notebook demonstrates how you can create a parameterized configuration JSON file that
# MAGIC will be later used for feature engineering, model training, and evaluation
# MAGIC (See the **run_ml_experiment_mlflow** notebook).
# MAGIC The intent is to create a reusable file containing the experimentation parameters
# MAGIC that data scientists can leverage,
# MAGIC with the goal of reducing overhead, promoting consistency in experimentation, and ensuring reproducibility.
# MAGIC 
# MAGIC **The JSON configuration file is used to specify parameters such as but not limited to**:
# MAGIC - The dataset that will be used for model training and evaluation.
# MAGIC - Important columns of the dataset to consider (like `target_colname` or `time_colname`).
# MAGIC - Feature engineering steps to take.
# MAGIC - The ML algorithm that will be used for the experiment.
# MAGIC - The cross-validation methodology to use for the experiment.
# MAGIC 
# MAGIC **Prerequisites**:
# MAGIC - This notebook is designed to consider only database tables that have been processed by existing ETL pipelines.
# MAGIC - It is assumed that database tables have already been tested and reconciled with source data by data engineers.
# MAGIC - Additionally, it is assumed that you have performed data understanding/validation, missing value imputations,
# MAGIC and Exploratory Data Analysis (EDA) prior to using this notebook.
# MAGIC Please be sure to verify your input data so that it is ready for model training and evaluation,
# MAGIC such as checking the join criteria, validating the date column format, checking the cross-validation ranges,
# MAGIC and ensuring that the config has the right feature engineering steps.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enable Arrow and Delta caching

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.databricks.io.cache.enabled", "true")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 0. Import libraries

# COMMAND ----------

import json
import sys
from pprint import pprint

# COMMAND ----------

# TSFA library imports:
sys.path.insert(0, '../..')
from tsfa.common.config_manager import ConfigManager

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1. Create configuration dictionary
# MAGIC 
# MAGIC Below is an example of the configuration dictionary which is user specified and drives the feature engineering and model training & evaluation workflow. In the following section, we provide a description of attributes of the config.

# COMMAND ----------

manager = ConfigManager(path_to_config="./json/exp_smoothing_config_small.json")
exp_smooth_config = manager.get()
pprint(exp_smooth_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Description of configuration file attributes
# MAGIC Below is a description of attributes of the configuration dictionary:
# MAGIC 
# MAGIC ```
# MAGIC "metadata": Metadata that highlights the market unit, product hierarchy and granularity we are working at.
# MAGIC {
# MAGIC     "time_series": The name of time series dataset (i.e. "sales").
# MAGIC },
# MAGIC "dataset": The database name and table name of the dataset to use.
# MAGIC {
# MAGIC     "db_name": The name of the database.
# MAGIC     "table_name": The name of the database table.
# MAGIC },
# MAGIC "dataset_schema": Specifies some important/named columns of the dataset, which are referenced by the framework for feature engineering, model training, etc.
# MAGIC {
# MAGIC     "required_columns": The list of columns of the database table that will be used to load the input Dataframe.
# MAGIC     "grain_colnames": The list of granularity columns of the time series (group_by columns).
# MAGIC     "time_colname": The name of the time column.
# MAGIC     "target_colname": The name of the target column that we want to forecast (i.e. revenue, volume, etc.).
# MAGIC     "forecast_colname": The name of the column to store model predictions as (i.e. forecast, y_hat, etc.).
# MAGIC     "ts_freq": The frequency of your time series. For example, use "W-SUN" to indicate weekly series every Sunday, or "MS" to indicate 1st of every month.
# MAGIC                See (https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases) for a list of supported frequency aliases.
# MAGIC },
# MAGIC "data_splitting": Attributes you need to create splits of your data.
# MAGIC {
# MAGIC     "holdout_timeframe": The start and end date of holdout time frame. If holdout_timeframe is not provided, a holdout set will not be computed.
# MAGIC                          (A holdout set is expected to be a slice of the data at the end, purely for evaluation and not for model training or validation)
# MAGIC     {
# MAGIC         "start": The start date of the holdout set (i.e."2021-08-01").
# MAGIC         "end": The end date of the hodout set (i.e. "2022-04-17").
# MAGIC     },
# MAGIC     "train_validation_timeframe": The start and end date of train and validation timeframe to compute time splits. 
# MAGIC                                   If train_validation_timeframe is not specified, train and validation sets will be
# MAGIC                                   extracted based on cross validation parameters and forecast horizon upto holdout_timeframe 
# MAGIC                                   start date [or] end of the dataframe (if holdout_timeframe is not provided)
# MAGIC     { 
# MAGIC         "start": The start date of the training set (i.e. "2015-01-01").
# MAGIC         "end": The end date of the validation set (i.e "2021-07-31").
# MAGIC     },
# MAGIC     "cross_validation": Rolling origin cross validation parameters that will be used to compute splits of the 
# MAGIC                         train_validation_timeframe. If cross_validation parameters are not specified, a simple 
# MAGIC                         split of train, validation and holdout will be provided based on train_validation_timeframe, 
# MAGIC                         holdout_timeframe and forecast_horizon. Refer to the Rolling Origin Validation document 
# MAGIC                         ('./docs/Model Experimentation and Evaluation Recommendations.md') for further details.
# MAGIC     {
# MAGIC         "num_splits": The number of rolling origin cross validation splits (i.e. 8).
# MAGIC         "rolling_origin_step_size": The rolling origin step size in time steps (i.e. 2).
# MAGIC     },
# MAGIC },
# MAGIC "feature_engineering": For exponential smoothing, this is not applicable.
# MAGIC "model_params": All parameters relevant for model training, such as the algorithm type and model hyperparameters.
# MAGIC {
# MAGIC     "ml_algorithm": The name of the algorithm to use.
# MAGIC     "hyperparameters": The hyperparameter of the ml_algorithm used. This field should be customized according to the 
# MAGIC                        ml_algorithm used. 
# MAGIC                        Below are some of the descriptions of model parameters-
# MAGIC                        * initialization_method: str, optional
# MAGIC                                                  Method for initialize the recursions.
# MAGIC                        * initial_level: float, optional
# MAGIC                                         The initial level component. Required if estimation method is “known”
# MAGIC                        * smoothing_level: float, optional
# MAGIC                                          The smoothing_level value of the simple exponential smoothing, 
# MAGIC                                          if the value is set then this value will be used as the value.
# MAGIC                        * optimized : bool, optional
# MAGIC                                      Estimate model parameters by maximizing the log-likelihood.
# MAGIC                        * start_params : ndarray, optional
# MAGIC                                      Starting values to used when optimizing the fit. If not provided, starting values are determined using a 
# MAGIC                                      combination of grid search and reasonable values based on the initial values of the data.
# MAGIC                        * use_brute : bool, optional
# MAGIC                                      Search for good starting values using a brute force (grid) optimizer. If False, a naive set of starting values is used.
# MAGIC                        Below is the link refrencing all Model parameters in Simple Exponential smoothing:
# MAGIC                        https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.SimpleExpSmoothing.html
# MAGIC     "model_name_prefix": The name of the model that will be registered to MLflow after appending with current date and time.
# MAGIC },
# MAGIC "forecast_horizon": The length of time steps (number of weeks, months, etc.) into the future for which forecasts are to be made.
# MAGIC                     This field is different than feature_horizon since it is a future period of time that the forecasts will be
# MAGIC                     produced and the two can have a different value. This field also represents the size of validation or 
# MAGIC                     holdout sets that get created when doing data splitting.
# MAGIC "results": {
# MAGIC     "db_name": The database name where the forecast outputs should be saved.
# MAGIC     "table_name": The table name where the forecast outputs should be saved.
# MAGIC }
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2. Save config to DBFS
# MAGIC Save the config dict as JSON files to DBFS.

# COMMAND ----------

# Config filename parameters:
# If you wish to define your own custom config filename, you can do so by modifying the arguments in the
# generate_file_path() function to `random = False` and defining a custom `filename_prefix`. For example,
# generate_file_path(random=False, filename_prefix="custom_123") would save the file as "custom_123.json".

filename_prefix = f'config_{exp_smooth_config["model_params"]["model_name_prefix"]}'
dbfs_directory = "test_configs"
filepath_to_save = manager.generate_filepath(path_to_folder=f"/dbfs/{dbfs_directory}", filename_prefix=filename_prefix, random=True)
pprint(filepath_to_save)

# COMMAND ----------

# Save Config to DBFS:
manager.save(config=exp_smooth_config, path_to_save=filepath_to_save)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate config is saved to DBFS

# COMMAND ----------

# Validate saved config:
with open(filepath_to_save, 'r') as f:
    final_config = json.load(f)

pprint(final_config)
