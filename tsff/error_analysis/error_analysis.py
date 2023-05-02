import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
from typing import Dict
from tsff.evaluation import WMapeEvaluator
from pyspark.sql import DataFrame as SparkDataFrame


class ErrorAnalysis:
    """Error Analysis class for time series forecasting"""

    def __init__(
        self,
        config: Dict
    ):
        """
        ML Experiment class constructor.

        Args:
            config (Dict): Configuration dictionary.
        """

        self.config = config
        self.target_column_name = config['dataset_schema']['target_colname']
        self.time_identifier = config['dataset_schema']['time_colname']
        self.keys_identifier = config['dataset_schema']['grain_colnames']
        self.predicted_column_name = config['dataset_schema']['forecast_colname']

    def cohort_plot(
        self,
        all_results: SparkDataFrame,
        walk_name: str = "walk",
        vmin: int = 30,
        vmax: int = 150,
    ) -> None:
        """
        df.columns = [self.time_identifier] + self.keys_identifier + [self.target_column_name, self.predicted_column_name, lookahead, walk_name]
        :param metric: string value from list of metrics [rmse, mape, mse] to get performing by.
        This function is ploting the metric score for cohort analysis
        """
        # get the data
        wmape = WMapeEvaluator()
        df_groupby = wmape.compute_metric_per_grain(df=all_results,
                                         target_colname=self.target_column_name,
                                         forecast_colname=self.predicted_column_name,
                                         grain_colnames=[walk_name, self.time_identifier])
        df_cohort = df_groupby.toPandas().sort_values(by=[walk_name, self.time_identifier])
        pivot_cohort = pd.pivot_table(
            df_cohort,
            values=['wmape'],
            index=[walk_name],
            columns=[self.time_identifier],
            aggfunc=np.mean,
        )
        # Initialize the figure
        plt.figure(figsize=(16, 10))
        # Adding a title
        plt.title(f"Average wmape: Weekly Cohorts", fontsize=14)
        # Creating the heatmap
        sns.heatmap(
            pivot_cohort.round(1),
            annot=True,
            vmin=vmin,
            vmax=vmax,
            cmap="YlGnBu",
            fmt="g",
        )
        plt.ylabel("walk_name")
        plt.xlabel(f"prediction {self.time_identifier}")
        plt.yticks(rotation="360")
        plt.show()

    def plot_time(self, df: SparkDataFrame) -> None:
        """
        df.columns = [self.time_identifier] + self.keys_identifier + [self.target_column_name, self.predicted_column_name]
        :param metric: string value from list of metrics [rmse, mape, mse] to get performing by.
        This function is ploting the metric score for each iteration per date
        """
        # get the data
        wmape = WMapeEvaluator()
        df_time = wmape.compute_metric_per_grain(df=df,
                                         target_colname=self.target_column_name,
                                         forecast_colname=self.predicted_column_name,
                                         grain_colnames=[self.time_identifier])
        df_rank = df_time.toPandas().sort_values(by=[self.time_identifier])
        fig, ax = plt.subplots(figsize=(15, 5))
        plt.xlabel(self.time_identifier)
        plt.ylabel('wmape')
        plt.title(f"wmape per iteration on dates")
        plt.plot(df_rank[self.time_identifier], df_rank['wmape'])
        ax.xaxis.set_tick_params(rotation=30, labelsize=10)
        plt.show()

    def plot_hist(
        self,
        df: SparkDataFrame,
        keys: list,
        bins=20,
        precentile=0.95,
        cut=False,
    ) -> None:
        """
        df.columns = [self.time_identifier] + self.keys_identifier + [self.target_column_name, self.predicted_column_name]
        :param metric: string value from list of metrics [rmse, mape, mse] to get performing by.
        :param bins: number of bins in the histograma.
        :param precentile: cut the edge of the data 0.95 => 95%.
        :param cut: Using pd.cut() to get a better understanding of the data.
        This function is ploting the distrubtion of the mape per product, location over time
        """
        wmape = WMapeEvaluator()
        df_groupby = wmape.compute_metric_per_grain(df=df,
                                         target_colname=self.target_column_name,
                                         forecast_colname=self.predicted_column_name,
                                         grain_colnames=keys)
        df_rank = df_groupby.toPandas()
        df_rank = df_rank[df_rank['wmape'] < df_rank['wmape'].quantile(precentile)]
        if cut:
            df_rank["bin"] = pd.cut(df_rank['wmape'], bins=bins).astype(str)
            df2 = df_rank.groupby("bin").bin.count()
            # Fixed to show distribution of bin
            df2.plot(kind="bar", xlabel='wmape', figsize=(15, 5))
            pl.suptitle(f"{' '.join(keys)} histogram")
        else:
            df_rank['wmape'].hist(bins=bins, legend=True, figsize=(15, 5))
            pl.suptitle(f"{' '.join(keys)} histogram")

    def plot_examples(
        self,
        df: SparkDataFrame,
        top=True,
        num_of_pairs=10
    ) -> None:
        """
        df.columns = [self.time_identifier] + self.keys_identifier + [self.target_column_name, self.predicted_column_name]
        :param df: Data Frame with the above columns
        :param top: Bool to get the top performing or False for the worst
        :param num_of_pairs: Int to get the amount of plots pairs self.keys_identifier (ex: sku, store)
        :param metric: string value from list of metrics [rmse, mape, mse] to get performing by.
        """

        wmape = WMapeEvaluator()
        df_groupby = wmape.compute_metric_per_grain(df=df,
                                         target_colname=self.target_column_name,
                                         forecast_colname=self.predicted_column_name,
                                         grain_colnames=self.keys_identifier)
        df_rank = df_groupby.toPandas()

        ls_wic_store_top = (
            df_rank.sort_values('wmape', ascending=top)
            .head(num_of_pairs)[self.keys_identifier + ['wmape']]
            .to_dict("records")
        )
        df = df.toPandas().sort_values(by=self.keys_identifier + [self.time_identifier])

        df["key_combine"] = ""
        for key in self.keys_identifier:
            df["key_combine"] = df["key_combine"] + '_' + df[key]
        for i, rec in enumerate(ls_wic_store_top):
            plot1 = plt.figure(i + 1)
            filter_str = ""
            for key in self.keys_identifier:
                filter_str += '_' + rec[key]
            df_loop = df[df["key_combine"] == filter_str].sort_values(by=[self.time_identifier])
            plt.xlabel(self.time_identifier)
            plt.ylabel(self.target_column_name)
            metric_value = round(rec['wmape'], 2)
            plt.title(f"{' '.join(self.keys_identifier)} : {filter_str[1:]}, {'wmape'} : {metric_value}")
            plt.plot(
                df_loop[self.time_identifier],
                df_loop[self.target_column_name],
                color="r",
                label=self.target_column_name,
            )
            plt.xticks(rotation=45)
            plt.plot(
                df_loop[self.time_identifier],
                df_loop[self.predicted_column_name],
                color="g",
                label=self.predicted_column_name,
            )
            plt.legend()
        plt.show()