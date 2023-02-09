# Databricks notebook source
# MAGIC %md
# MAGIC # Using ConfigManager
# MAGIC This notebook shows the usage of the ConfigManager class to load, save and modify a config file.

# COMMAND ----------

from pprint import pprint

import sys
sys.path.insert(0, '../..')
from tsff.common.config_manager import ConfigManager

# COMMAND ----------

config_manager = ConfigManager(path_to_config='../configs/json/random_forest_config_small.json')
config = config_manager.get()
pprint(config)

# COMMAND ----------

config_num_splits_change = config_manager.modify(path_to_section=['data_splitting', 'cross_validation'], new_section_values={'num_splits': 4}, in_place=False)
pprint(config_num_splits_change)

# COMMAND ----------

config_manager.save(config_num_splits_change, path_to_save="../test_config.json")
!ls ../*.json

# COMMAND ----------

!rm ../test_config.json
