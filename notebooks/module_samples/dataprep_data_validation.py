# Databricks notebook source
# MAGIC %md
# MAGIC # Data Preparation: Data validation
# MAGIC This notebook demonstrates the different data validation steps that take place throughout TSFA, such as checking pandas vs spark dataframes, checking for duplicate rows, checking nulls, checking expected column names of data that was loaded, and data leakage between splits of the dataframe given we are working with time series.

# COMMAND ----------

import pandas as pd
from pprint import pprint

import sys
sys.path.insert(0, '../..')
from tsfa.data_prep.data_validation import *

# COMMAND ----------

# Spark dataframe
df_sp = spark.read.csv("dbfs:/FileStore/tables/dominicks_oj_small.csv", header=True)

# COMMAND ----------

# Pandas dataframe
df_pd = df_sp.toPandas()
df_pd['date'] = pd.to_datetime(df_pd['date'])
df_pd.head()

# COMMAND ----------

# Another toy pandas dataframe with nulls
toy_df = pd.DataFrame({
  'a': [1, 2],
  'b': [3, None]
})
toy_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check dataframe type

# COMMAND ----------

check_df_type(df_pd)
check_df_type(df_sp)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check nulls

# COMMAND ----------

# Check for nulls in oj data. Should raise an error and print how many nulls we have per column
check_dataframe_nulls(df_pd)

# COMMAND ----------

# Check for nulls in toy_df: SHould raise an error
check_dataframe_nulls(toy_df)

# COMMAND ----------

# Drop nulls from orange juice dataframe and check for nulls again with method
df_pd_no_nulls = df_pd.dropna()
# Check for nulls in df_pd_no_nulls: SHould not raise an error
check_dataframe_nulls(df_pd_no_nulls)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Check column names

# COMMAND ----------

# Same  number of columns but differ in one instance: Should raise an error
check_dataframe_colnames(df_colnames=['a', 'b', 'c'], expected_colnames=['a', 'b', 'd'])

# COMMAND ----------

# Different number of columns: Should raise an error
check_dataframe_colnames(df_colnames=['a', 'b'], expected_colnames=['a', 'b', 'd'])

# COMMAND ----------

# Should raise an error
check_dataframe_colnames(df_colnames=['a', 'b', 'b'], expected_colnames=['a', 'b', 'd'])

# COMMAND ----------

# Should raise an error
check_dataframe_colnames(df_colnames={'a', 'b', 'b'}, expected_colnames=['a', 'b', 'd'])

# COMMAND ----------

# Shouldn't raise an error
check_dataframe_colnames(df_colnames=['a', 'b', 'c'], expected_colnames=['a', 'b', 'c'])

# COMMAND ----------

# Permuted order of columns: Shouldn't raise an error
check_dataframe_colnames(df_colnames=['a', 'b', 'c'], expected_colnames=['c', 'a', 'b'])

# COMMAND ----------

# Should raise an error
check_dataframe_colnames(df_colnames=['a', 'b', 'c', 'b'], expected_colnames=['c', 'c', 'a', 'b'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Duplicate rows check

# COMMAND ----------

# There should be no duplicates
check_duplicate_rows(df_pd, colnames=df_pd.columns)

# COMMAND ----------

# There should be duplicate rows with the pandas dataframe
check_duplicate_rows(df_pd, colnames=['store', 'brand'])

# COMMAND ----------

# There should be no duplicates with the spark dataframe
check_duplicate_rows(df_sp, colnames=df_sp.columns)

# COMMAND ----------

# There should be duplicate rows with the spark dataframe when we restrict our attention to a few columns
check_duplicate_rows(df_sp, colnames=['store', 'brand'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time splits JSON configuration validation

# COMMAND ----------

splits_json_data = {
  "holdout": {
    "start": "2022-01-01",
    "end": "2022-04-01"
  },
  "validation": {
    "split1": {
      "end": "2021-12-31",
      "start": "2021-09-30"
    }
  }
}


# COMMAND ----------

# Should raise an error
data_bad_no_train_key = splits_json_data.copy()
validate_splits_json_config(data_bad_no_train_key)

# COMMAND ----------

# Should raise an error
data_bad_train_misspelt = splits_json_data.copy()
data_bad_train_misspelt["trainng"] = {
    "split1": {
      "start": "2021-01-01",
      "end": "2021-09-29"
    }
}
validate_splits_json_config(data_bad_train_misspelt)

# COMMAND ----------

# Should raise an error
data_bad_train_key_null = splits_json_data.copy()
data_bad_train_key_null["training"] = None
validate_splits_json_config(data_bad_train_key_null)

# COMMAND ----------

# Should raise an error
data_bad_train_split1_key_null = splits_json_data.copy()
data_bad_train_split1_key_null["training"] = {
    "split1": None
}
validate_splits_json_config(data_bad_train_split1_key_null)

# COMMAND ----------

# Should raise an error
data_bad_train_split1_start_key_null = splits_json_data.copy()
data_bad_train_split1_start_key_null["training"] = {
    "split1": {
      "start": None,
      "end": "2021-09-29"
    }
}
validate_splits_json_config(data_bad_train_split1_start_key_null)

# COMMAND ----------

# Should raise an error
data_bad_train_split1_start_key_not_string = splits_json_data.copy()
data_bad_train_split1_start_key_not_string["training"] = {
    "split1": {
      "start": 42,
      "end": "2021-09-29"
    }
}
validate_splits_json_config(data_bad_train_split1_start_key_not_string)

# COMMAND ----------

# Should raise an error
data_bad_date_format = splits_json_data.copy()
data_bad_date_format["training"] = {
    "split1": {
      "start": "01-01-2021",
      "end": "2021-10-01"
    }
}
validate_splits_json_config(data_bad_date_format)

# COMMAND ----------

# Should raise an error
data_bad_start_later_than_end = splits_json_data.copy()
data_bad_start_later_than_end["training"] = {
    "split1": {
      "start": "2021-10-02",
      "end": "2021-10-01"
    }
}
validate_splits_json_config(data_bad_start_later_than_end)

# COMMAND ----------

# Finally passing in a kson with the right format
splits_json_right_format = splits_json_data.copy()
splits_json_right_format["training"] = {
    "split1": {
      "start": "2021-01-01",
      "end": "2021-10-01"
    }
}
validate_splits_json_config(splits_json_right_format)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Data leakage

# COMMAND ----------

pprint(splits_json_right_format)

# COMMAND ----------

# Should raise an error due to leakage detection
check_data_leakage(splits_json_right_format)

# COMMAND ----------

# Fix leakage, shouldn't have error
splits_json_no_leak = splits_json_data.copy()
splits_json_no_leak["training"] = {
    "split1": {
      "start": "2021-01-01",
      "end": "2021-09-29"
    }
}
check_data_leakage(splits_json_no_leak)
