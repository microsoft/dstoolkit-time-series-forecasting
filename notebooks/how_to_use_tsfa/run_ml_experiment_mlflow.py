# Databricks notebook source
# MAGIC %md
# MAGIC # Run ML Experiment and Log to MLFlow
# MAGIC ### Overview
# MAGIC 
# MAGIC This notebook demonstrates how to use TSFA end-to-end to run a ML experiment. The objective of this notebook is to provide a reusable framework for model training and evaluation. The specific parameters of the experiment is provided in a JSON config file, which can be created using the **create_*_config** notebook for your model type (e.g. `create_random_forest_config`). This notebook will use the JSON config to read the input DataFrame, compute features/transformations, train the specified ML model, and evaluate model performance using walk-forward cross validation. Additionally, for traceability, re-producability, and ease of comparison, the model artifacts produced will be logged to MLFlow.
# MAGIC 
# MAGIC ### Pre-requisites
# MAGIC 
# MAGIC In order to run this notebook, you will need:
# MAGIC   1. The input dataset, as a Spark DataFrame, which will be used to train and evaluate your model.
# MAGIC   2. The JSON config file which contains the parameters of the run as well as the CV time splits. Refer to the demo notebooks that illustrate expected config schema for details
# MAGIC 
# MAGIC Additionally, it is assumed that you have performed data understanding/validation, missing value imputations, and Exploratory Data Analysis (EDA) prior to using this notebook. Please be sure to verify your input data so that it is ready for model training and evaluation, such as checking the join criteria, validating the date column format, checking the cross-validation ranges, and ensuring that the config has the right feature engineering steps.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enable Arrow and Delta caching

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.databricks.io.cache.enabled", "true")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Libraries

# COMMAND ----------

import json
import mlflow
import mlflow.sklearn
import sys
import time
from datetime import datetime
from mlflow.tracking import MlflowClient
from pprint import pprint
from typing import Dict, Union

# COMMAND ----------

# TSFA library imports:
sys.path.insert(0, '../..')
from tsfa.common.config_manager import ConfigManager
from tsfa.data_prep.data_prep_utils import DataPrepUtils
from tsfa.models import RandomForestRegressorModel, ProphetModel, RollingMeanModel, SimpleExpSmoothingModel
from tsfa.ml_experiment import MLExperiment
from tsfa.evaluation import __name__ as evaluation_module_name, BaseEvaluator, CompoundEvaluator

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load and parse Json config
# MAGIC The Json config will be used to specify:
# MAGIC  - The input datasets
# MAGIC  - ML algorithm to use
# MAGIC  - Feature engineering/transformations to conduct on the dataset
# MAGIC  - Model hyperparameters to use
# MAGIC  - Cross-validation date ranges when running walk-forward CV
# MAGIC  - Metadata for file naming convention

# COMMAND ----------

# Configuration file to use:
config_filename = "prophet_config_small.json" # "YOUR_CONFIG_JSON_FILE"
config_path = f"../configs/json/{config_filename}"

# ConfigParser contains helper methods to read in the contents a Config file. Future versions will also have
# methods to validate the contents of the config.
cnf_manager = ConfigManager(config_path)
config = cnf_manager.get()
pprint(config)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialize data prep utils and load dataset
# MAGIC
# MAGIC To load the data successfully using the default `prophet_config_small.json` config file, please ensure the **`data/dominicks_oj_data/create_oj_data_small.py` notebook is executed successfully**. The notebook will create the database and table required for this notebook.

# COMMAND ----------

data_prep_utils = DataPrepUtils(spark, config)

# COMMAND ----------

# Read in data (as Spark Dataframe):
df = data_prep_utils.load_data(as_pandas=False)
print(df.count(), '|', len(df.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Instantiate the Evaluator class and the MLExperiment class
# MAGIC 
# MAGIC MLExperiment provides the flexibility for a user to specify in a simple way any model they wish to train and the metric they want to evaluate against.
# MAGIC 
# MAGIC In the instantiation below, we have
# MAGIC ```
# MAGIC ml_exp = MLExperiment(spark_session=spark,
# MAGIC                       config=config,
# MAGIC                       model_cls=RollingMeanModel,
# MAGIC                       evaluator_obj=WMapeEvaluator())
# MAGIC ```
# MAGIC 
# MAGIC The `config` parameter takes the driver configuration file that we load at the top of this notebook. `RollingMeanModel` is a class object that gets instantiated within MLExperiment. Note that the `config` has a `model_params` field that specifies which model the user wants to train with needed parameters. The `MLExperiment` class checks if the model specified in the `config` aligns with the model class that the user wants to leverage. If these are different, an exception is raised. As long as these parameters align, the user can pass any other model class object here such as `RandomForestRegressrModel` and the associated config that specifies parameters for random forest.

# COMMAND ----------

def build_evaluators(config: Dict) -> Union[BaseEvaluator, CompoundEvaluator]:
    """
    Build evaluator object from the config specification.
    
    Args:
        config (Dict): Configuration dictionary.
    
    Returns:
        evaluator (Union[BaseEvaluator, CompoundEvaluator]): Instance of an evaluator or a compound evaluator.
    """
    # Check whether evaluation is specified in config
    if ("evaluation" not in config) or (config["evaluation"] is None) or (len(config["evaluation"]) == 0):
        return None
    # Build evaluator objects as described in the config
    evaluators_list = []
    for e in config["evaluation"]:
        # Get evaluator class
        evaluator_cls = getattr(sys.modules[evaluation_module_name], e["metric"])
        # Instantiate evaluator using specified parameters
        single_evaluator = evaluator_cls(**{k: v for k, v in e.items() if k != "metric"})
        evaluators_list += [single_evaluator]
    # Use CompoundEvaluator if many are specified
    evaluator = CompoundEvaluator(evaluators_list) if len(evaluators_list) > 1 else evaluators_list[0]
    return evaluator

# COMMAND ----------

# Instantiate evaluation metric:
evaluator = build_evaluators(config)

# Instantiate ML Experiment class:
ml_exp = MLExperiment(spark_session=spark,
                      config=config,
                      model_cls=ProphetModel,
                      evaluator_obj=evaluator)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run walk-forward model training and log experiment to MLFlow
# MAGIC This step runs the model training and evaluation. It will log the important run information to MLFlow so that it can be referenced and reviewed later:
# MAGIC   - Dataset used
# MAGIC   - Target column used
# MAGIC   - CV time splits
# MAGIC   - The mean metrics from walk-forward CV
# MAGIC   - The standard deviation of the metrics from walk-forward CV
# MAGIC   - The total runtime (in seconds)
# MAGIC   - The config file used for the run
# MAGIC 
# MAGIC By default, MLExperiment will perform k-fold walk forward model training (`ml_exp.walk_forward_model_training()`). Data splits will be automatically created using the `data_splitting` parameters provided in the config. If you wish to override the automatic data splitting and input a custom data splitting scheme, the following dictionary can be used as an example:
# MAGIC 
# MAGIC     custom_time_splits = {
# MAGIC         'holdout': {'end': '2022-04-17', 'start': '2021-08-01'},
# MAGIC         'training': {'split1': {'end': '2020-11-01', 'start': '2015-01-01'},
# MAGIC                      'split2': {'end': '2020-10-18', 'start': '2015-01-01'},
# MAGIC                      'split3': {'end': '2020-10-04', 'start': '2015-01-01'},
# MAGIC                      'split4': {'end': '2020-09-20', 'start': '2015-01-01'}},
# MAGIC         'validation': {'split1': {'end': '2021-07-25', 'start': '2020-11-08'},
# MAGIC                        'split2': {'end': '2021-07-11', 'start': '2020-10-25'},
# MAGIC                        'split3': {'end': '2021-06-27', 'start': '2020-10-11'},
# MAGIC                        'split4': {'end': '2021-06-13', 'start': '2020-09-27'}}
# MAGIC     }
# MAGIC 
# MAGIC     walk_forward_results = ml_exp.walk_forward_model_training(df, time_splits=custom_time_splits)

# COMMAND ----------

model_name = f'{config["model_params"]["model_name_prefix"]}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
with mlflow.start_run(run_name=f"run_{model_name}") as run:
    # Start run:
    run_id = run.info.run_id
    print("Current run_id: {}".format(run_id))

    # Run walk-forward model training. If the `time_splits` argument is None, MLExperiment will create time
    # splits automatically from the config's 'data_splitting' parameters. Otherwise, `time_splits` can be
    # manually set.
    t0 = time.time()
    walk_forward_results = ml_exp.walk_forward_model_training(df, time_splits=None, verbose=False)
    rt = time.time() - t0

    # Log metrics:
    print('Experiment runtime:', rt)
    print('Evaluators mean: ', walk_forward_results.avg_metrics)
    print('Evaluators std_dev: ', walk_forward_results.std_metrics)
    mlflow.log_param("dataset", config['dataset'])
    mlflow.log_param("target_colname", config['dataset_schema']['target_colname'])
    mlflow.log_param("forecast_colname", config['dataset_schema']['forecast_colname'])
    mlflow.log_param("time_colname", config['dataset_schema']['time_colname'])
    mlflow.log_param("grain_colnames", config['dataset_schema']['grain_colnames'])
    mlflow.log_param("model_params", str(config['model_params']))
    mlflow.log_metric("mean_wmape", walk_forward_results.avg_metrics['wmape'])
    mlflow.log_metric("std_wmape", walk_forward_results.std_metrics['wmape'])
    mlflow.log_metric("runtime_in_seconds", rt)

    # Log config file:
    mlflow.log_artifact(config_path, artifact_path="config_json_path")

# COMMAND ----------

# Show all walk forward run results for visibility (if needed):
all_metrics = {i.run_name:i.metrics.collect() for i in walk_forward_results.run_results}
pprint(all_metrics)
