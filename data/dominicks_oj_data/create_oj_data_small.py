# Databricks notebook source
# MAGIC %md
# MAGIC ### Creating a small sample of orange juice dataset that is rectangular
# MAGIC 
# MAGIC We work with the Dominicks orange juice dataset. The focus for this notebook is to retain only those brand in each store that have features and target values for all the dates ranging from `1990-06-20` till `1992-10-07`.

# COMMAND ----------

import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read in OJ data and save subset

# COMMAND ----------

# Loading a small sample of dominicks orange juice dataset that has staggered data

df = pd.read_csv("dominicks_oj_staggered.csv", header=0)
print(df.shape)
df.head()

# COMMAND ----------

# Counting number of dates for each grain (store, brand): 
# Shows many grains have information for the complete 
# set of 121 dates whereas there are some grains that do not
count_dates_per_grain = (
    df
    .groupby(['brand', 'store'])
    .agg({'date': 'count'})
    .rename(columns={'date': 'count'})
    .reset_index()
    )
count_dates_per_grain

# COMMAND ----------

# Retaining only those grains that have complete information for the length of the timeframe
grains_less = count_dates_per_grain[count_dates_per_grain['count'] == 121].drop(columns='count')

# COMMAND ----------

# Creating a sample with just the filtered set of grains
df_sample = grains_less.merge(df, on=['brand', 'store'], how='left')
print(df_sample.shape)
df_sample.head()

# COMMAND ----------

# Visualizing a single series
trop_samp = df_sample[(df_sample['brand'] == 'tropicana') & (df_sample['store'] == 2)].copy()
trop_samp.sort_values(by = 'date', inplace = True)
trop_samp

# COMMAND ----------

# Modifications to prepare to save as Spark df
df_sample['brand'] = np.where(df_sample['brand'] == 'minute.maid', 'minute_maid', df_sample['brand'])
df_sample['date'] = pd.to_datetime(df_sample['date'])
df_sample = df_sample[['date', 'brand', 'store', 'on_promotion', 'price', 'quantity']]
df_sample.reset_index(drop = True, inplace = True)

# COMMAND ----------

# Fix some NaNs in quantity
df_sample['quantity'].fillna(0, inplace = True)

# COMMAND ----------

# Saving the sample back to the data folder as csv
df_sample.to_csv("dominicks_oj_small.csv", index = False)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create pyspark df and save to database

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, FloatType, StringType, DateType

# COMMAND ----------

# Dataframe schema
schema = StructType([
    StructField("date", DateType(), True),
    StructField("brand", StringType(), True),
    StructField("store", StringType(), True),
    StructField("on_promotion", FloatType(), True),
    StructField("price", FloatType(), True),
    StructField("quantity", FloatType(), True)
])

# COMMAND ----------

# Save df
df_sample_spark = spark.createDataFrame(df_sample, schema=schema)
df_sample_spark.write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable("sample_data.orange_juice_small")
