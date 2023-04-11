import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

class ErrorAnalysis:
    def __init__(
        self,
        target_column_name: str = "actual",
        time_identifier: str = "week",
        keys_identifier: list = ["key1", "key2"],
        predicted_column_name: str = "prediction"
    ):
        """
        :param target_column_name: Name of column containing the target
        :param time_identifier: Name of column containing the date
        :param keys_identifier: List of the keys
        :param predicted_column_name : Column name representing the prediction results
        """
        self.target_column_name = target_column_name
        self.time_identifier = time_identifier
        self.keys_identifier = keys_identifier
        self.predicted_column_name = predicted_column_name

    def _evaluator_evaluate(self, df):
        y_actual = df[self.target_column_name]
        y_predicted = df[self.predicted_column_name]
        mape = mean_absolute_error(y_actual, y_predicted)
        mse = mean_squared_error(y_actual, y_predicted)
        rmse = math.sqrt(mse)
        return {"mape": mape, "mse": mse, "rmse": rmse}

    def _run_evaluate(self, df):
        # this function is a helper function to split the df dataframe
        evaluate_results = self._evaluator_evaluate(df)
        return pd.Series(
            dict(
                mse=evaluate_results["mse"],
                rmse=evaluate_results["rmse"],
                mape=evaluate_results["mape"],
                min_date=min(df[self.time_identifier]),
            )
        )

    def get_metric_values(self, df: pd.DataFrame, keys) -> pd.DataFrame:
        """
        df.columns = [self.time_identifier] + self.keys_identifier + [self.target_column_name, self.predicted_column_name]
        It drop na values and get a pivot table of keys : self.keys_identifier (ex: sku, store)
                                                 values : [rmse, mape, mse]
        The use of this function is to rank the top performing or worst performing by the value matric.
        """
        df = df[~df[self.target_column_name].isna()]
        return df.groupby(keys, as_index=False).apply(self._run_evaluate)

    def cohort_plot(
        self,
        all_results: pd.DataFrame,
        walk_name: str = "walk",
        metric: str = "mape",
        vmin: int = 30,
        vmax: int = 150,
    ) -> None:
        """
        df.columns = [self.time_identifier] + self.keys_identifier + [self.target_column_name, self.predicted_column_name, lookahead, walk_name]
        :param metric: string value from list of metrics [rmse, mape, mse] to get performing by.
        This function is ploting the metric score for cohort analysis
        """
        # get the data
        df_cohort = self.get_metric_values(
            df=all_results, keys=[walk_name, self.time_identifier]
        )

        pivot_cohort = pd.pivot_table(
            df_cohort,
            values=[metric],
            index=[walk_name],
            columns=[self.time_identifier],
            aggfunc=np.mean,
        )
        # Initialize the figure
        plt.figure(figsize=(16, 10))
        # Adding a title
        plt.title(f"Average {metric}: Weekly Cohorts", fontsize=14)
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

    def plot_time(self, df: pd.DataFrame, metric="mape") -> None:
        """
        df.columns = [self.time_identifier] + self.keys_identifier + [self.target_column_name, self.predicted_column_name]
        :param metric: string value from list of metrics [rmse, mape, mse] to get performing by.
        This function is ploting the metric score for each iteration per date
        """
        # get the data
        df_rank = self.get_metric_values(df=df, keys=self.time_identifier)

        fig, ax = plt.subplots(figsize=(15, 5))
        plt.xlabel(self.time_identifier)
        plt.ylabel(metric)
        plt.title(f"{metric} per iteration on dates")
        plt.plot(df_rank[self.time_identifier], df_rank[metric])
        ax.xaxis.set_tick_params(rotation=30, labelsize=10)
        plt.show()

    def plot_hist(
        self,
        df: pd.DataFrame,
        keys: list,
        metric="rmse",
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
        df_rank = self.get_metric_values(df=df, keys=keys)
        df_rank = df_rank[df_rank[metric] < df_rank[metric].quantile(precentile)]
        if cut:
            df_rank["bin"] = pd.cut(df_rank[metric], bins=bins).astype(str)
            df2 = df_rank.groupby("bin").bin.count()
            # Fixed to show distribution of bin
            df2.plot(kind="bar", xlabel=metric, figsize=(15, 5))
            pl.suptitle(f"{' '.join(keys)} histogram")
        else:
            df_rank[metric].hist(bins=bins, legend=True, figsize=(15, 5))
            pl.suptitle(f"{' '.join(keys)} histogram")

    def plot_examples(
        self,
        df: pd.DataFrame,
        top=True,
        num_of_pairs=10,
        metric: str = "rmse",
    ) -> None:
        """
        df.columns = [self.time_identifier] + self.keys_identifier + [self.target_column_name, self.predicted_column_name]
        :param df: Data Frame with the above columns
        :param top: Bool to get the top performing or False for the worst
        :param num_of_pairs: Int to get the amount of plots pairs self.keys_identifier (ex: sku, store)
        :param metric: string value from list of metrics [rmse, mape, mse] to get performing by.
        """

        df_rank = self.get_metric_values(df=df, keys=self.keys_identifier)
        ls_wic_store_top = (
            df_rank.sort_values(metric, ascending=top)
            .head(num_of_pairs)[self.keys_identifier + [metric]]
            .to_dict("records")
        )
        df["key_combine"] = ""
        for key in self.keys_identifier:
            df["key_combine"] = df["key_combine"] + '_' + df[key]
        for i, rec in enumerate(ls_wic_store_top):
            plot1 = plt.figure(i + 1)
            filter_str = ""
            for key in self.keys_identifier:
                filter_str += '_' + rec[key]
            df_loop = df[df["key_combine"] == filter_str]
            plt.xlabel(self.time_identifier)
            plt.ylabel(self.target_column_name)
            metric_value = round(rec[metric], 2)
            plt.title(f"{' '.join(self.keys_identifier)} : {filter_str[1:]}, {metric} : {metric_value}")
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