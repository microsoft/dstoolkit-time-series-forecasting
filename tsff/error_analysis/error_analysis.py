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
        This function is ploting the metric score for cohort analysis
        The cohort plot is a plot of the error as a function of the time and the walk number
        the vmin and vmax are for the colorbar

        Args:
            all_results (SparkDataFrame): Spark dataframe with the results of the walk forward
            walk_name (str, optional): The name of the walk column. Defaults to "walk".
            vmin (int, optional): The minimum value for the colorbar. Defaults to 30.
            vmax (int, optional): The maximum value for the colorbar. Defaults to 150.

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
        This function is ploting the metric score for each iteration per date

        Args:
            df (SparkDataFrame): Spark dataframe with the results of the walk forward
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
        This function is ploting the distrubtion of the wmape per keys over time

        Args:
            df (SparkDataFrame): Spark dataframe with the results of the walk forward
            keys (list): list of the keys to group by
            bins (int, optional): number of bins in the histograma. Defaults to 20.
            precentile (float, optional): cut the edge of the data 0.95 => 95%. Defaults to 0.95.
            cut (bool, optional): Using pd.cut() to get a better understanding of the data. Defaults to False.
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
        This function is ploting the best or worst examples

        Args:
            df (SparkDataFrame): Spark dataframe with the results of the walk forward
            top (bool, optional): True for the best examples and False for the worst. Defaults to True.
            num_of_pairs (int, optional): number of examples to show. Defaults to 10.
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