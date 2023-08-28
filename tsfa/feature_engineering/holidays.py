"""
This is the DaysToHolidays submodule that is used by features.py to compute days to major
holidays and include them as features.
"""

import pandas as pd
import datetime as dt
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
import json


class DaysToHolidays:
    """This class computes days to holidays for each date in the dataframe based on a holidays dictionary."""

    def __init__(self,
                 spark_session: SparkSession,
                 time_colname: str,
                 holidays_path: str,
                 lower_bound: int = 56,
                 upper_bound: int = 14,
                 constant: int = 500):
        """
        Constructor for days to holidays featurizer.

        Args:
            spark_session (SparkSession): Existing spark session
            time_colname (str): The name of the time column to compute days to holidays features.
            holidays_path (str): The path to the json file for major holidays in DBFS. The json file has a dictionary
                                 structure with each holiday (i.e. "easter", "superbowl",...,"christmas") as keys and
                                 values as a list of specific holiday dates over the period of consideration
                                 (i.e. "christmas": ["2015-12-25", "2016-12-25", "2017-12-25", ...]).
            lower_bound (int): The lower bound for number of days before holidays.
            upper_bound (int): The upper bound for number of days after holidays.
            constant (int): A constant value to set the number of days to holidays that are outside of the range
                            (-lower_bound, upper_bound).
        """
        self.spark = spark_session
        self.time_colname = time_colname
        self.holidays_path = holidays_path
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.constant = constant
        self.load_holidays()

    def transform(self, df: SparkDataFrame) -> SparkDataFrame:
        """
        Computes days to holidays features for the input spark dataframe.

        Args:
            df (SparkDataFrame): The spark dataframe for calculating days to holidays.

        Returns:
            df_with_holidays (SparkDataFrame): The spark dataframe with days to holidays features.

        """
        # extract dates from the spark df and convert to pandas dataframe for processing.
        # Note: Since the time series periods are weekly or monthly, this results in a small dataframe.
        # Even with 20 years of data with weekly frequency, this will result in a dataframe with around 1000 rows.
        dates_pd = df.select(self.time_colname).distinct().toPandas()
        dates_pd[self.time_colname] = pd.to_datetime(dates_pd[self.time_colname], format="%Y-%m-%d")
        dates_pd.sort_values(self.time_colname, inplace=True)
        dates_pd.reset_index(drop=True, inplace=True)

        # compute days to holidays
        dates_with_holidays_pd = self.days_to_holidays(dates_pd)

        # feature columns we get after creating holiday features
        self._feature_colnames = [col for col in dates_with_holidays_pd.columns if col != self.time_colname]
        df_feats = self.spark.createDataFrame(dates_with_holidays_pd,
                                              schema=dates_with_holidays_pd.columns.tolist())

        # Merge results into input dataframe:
        df_result = df.join(df_feats, on=self.time_colname, how='left')
        return df_result

    def load_holidays(self):
        """Reads and type casts holidays json."""
        # load holidays
        with open(self.holidays_path, 'r') as f:
            holidays_json = f.read()
        holidays = json.loads(holidays_json)

        # convert string dates to datetime
        for holiday, value in holidays.items():
            ds = value['ds']
            holiday_dates = [dt.datetime.strptime(x, '%Y-%m-%d') for x in ds]
            holidays[holiday] = holiday_dates
        self.holidays = holidays

    def days_to_holidays(self, dates_pd: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates number of days to major holidays. We set a lower bound and an upper bound
        for the number of days that matter as a feature.
           - For example, consider Christmas, we want to capture all weeks that are 8 weeks (56 days) before Christmas
             (lower bound) and 2 weeks (14 days) after Christmas (upper bound) as crucial weeks that matter for revenue
             forecasting.
           - If the number of days to a holiday are outside of these bounds (-56, 14), we set to a arbitrary large
             constant (say, 500). If the days to holiday is larger than 8 weeks before a holiday, we set it to 500, and
             if the days to holiday is greater than 2 weeks after a holiday, we will also set this to 500.
        Args:
            dates_pd (pd.DataFrame): Dataframe of distinct dates in the time series.

        Returns:
            dates_with_holidays_pd (pd.DataFrame): Dataframe with days to each holiday.
        """
        dates_with_holidays_pd = dates_pd.copy()

        # loop through all holidays in the holidays dict (i.e. easter, superbowl,..., christmas)
        for holiday, holiday_dates in self.holidays.items():
            # create dataframe for holiday dates
            holiday_dates_df = pd.DataFrame(holiday_dates, columns=["holiday_dates"])
            # cross merge and compute difference in days between all dates and holiday dates
            days_to_holiday = dates_pd.merge(holiday_dates_df, how="cross")
            days_to_holiday["days_diff"] = (
                (days_to_holiday[self.time_colname] - days_to_holiday["holiday_dates"]).dt.days
            )
            # set days to holiday to constant if day difference is outside of (-lower_bound, upper_bound)
            days_to_holiday.loc[((days_to_holiday["days_diff"] < -self.lower_bound)
                                | (days_to_holiday["days_diff"] > self.upper_bound)),
                                "days_diff"] = self.constant
            # for each date, pick the day difference to the closest holiday date (min with ordering function as abs).
            dates_with_holidays_pd['days_to_' + holiday] = days_to_holiday.groupby(self.time_colname) \
                .apply(lambda x: min(x["days_diff"], key=abs)).values

        return dates_with_holidays_pd
