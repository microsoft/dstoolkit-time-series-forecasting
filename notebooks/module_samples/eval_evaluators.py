# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluators Sample Notebook
# MAGIC This notebook demonstrates how to call different evaluators (such as WMapeEvaluator, OffsetErrEvaluator, or CompoundEvaluator) for your workflows.

# COMMAND ----------

import sys
sys.path.insert(0, '../..')
from tsfa.common.dataloader import DataLoader
from tsfa.evaluation import CompoundEvaluator, WMapeEvaluator
from pprint import pprint
from pyspark.sql import functions as f

# COMMAND ----------

# Load test data:
loader = DataLoader(spark)
df = loader.read_df_from_mount(db_name="sample_data",
                               table_name="orange_juice_small",
                               as_pandas=False)
display(df)

# COMMAND ----------

df_with_forecasts = df.withColumn("forecasts", f.col('quantity') + f.lit(1))
display(df_with_forecasts)

# COMMAND ----------

# MAGIC %md
# MAGIC ### WMAPE Evaluator

# COMMAND ----------

# WMAPE, no grain columns:
wmape = WMapeEvaluator()
df_perf = wmape.compute_metric_per_grain(df=df_with_forecasts, 
                                        target_colname='quantity', 
                                        forecast_colname='forecasts', 
                                        grain_colnames=[])
df_perf.collect()

# COMMAND ----------

# With grain cols specified:
df_perf = wmape.compute_metric_per_grain(df=df_with_forecasts,
                                         target_colname='quantity',
                                         forecast_colname='forecasts',
                                         grain_colnames=['brand', 'store'])
display(df_perf)

# COMMAND ----------

# Express as accuracy:
wmape = WMapeEvaluator(express_as_accuracy = True)
df_perf = wmape.compute_metric_per_grain(df=df_with_forecasts,
                                         target_colname='quantity',
                                         forecast_colname='forecasts',
                                         grain_colnames=['store'])
display(df_perf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compound Evaluator

# COMMAND ----------

# No grain columns:
compeval = CompoundEvaluator(
    [
        WMapeEvaluator(),
        WMapeEvaluator(express_as_accuracy = True, metric_colname='wmape_acc')
    ]
)
df_perf = compeval.compute_metric_per_grain(df=df_with_forecasts, 
                                            target_colname='quantity', 
                                            forecast_colname='forecasts', 
                                            grain_colnames=['store', 'brand'])
display(df_perf)
