"""
This module implements a generic MLExperiment class, that orchestrates the
training, forecasting and evaluation of a time series model on some data.
"""

import multiprocessing as mp
from statistics import mean, stdev
from typing import Dict, Type, Union, Optional, List
from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
import time

from tsff.data_prep.data_prep_utils import DataPrepUtils
from tsff.evaluation.compound_evaluator import CompoundEvaluator
from tsff.feature_engineering.features import FeaturesUtils
from tsff.models.base_model import BaseModel
from tsff.evaluation import BaseEvaluator
from dataclasses import dataclass


@dataclass
class SingleMLRunResults:
    """
    Class to capture results of a single run of training, testing and evaluation

    Attributes:
        train_timeframe (Dict): The timeframe to train on. Format should look like this:
                                {"start": 2018-01-01, "end": "2020-12-31"}
        test_timeframe (Dict): The timeframe to test on. Format should look like this:
                               {"start": 2021-01-01, "end": "2021-12-31"}
        run_name (Optional[str]): The name for the run. This is useful for collecting the results of multiple
                                  runs together. For example, walk_forward_model_training() uses this to record
                                  the split names: ("Split1", "Split2", etc.).
        result_df (SparkDataFrame): Dataframe with actual targets and output forecasts for the test timeframe.
                                    The dataframe may also include specified granularity columns and any other
                                    information pertinent to the experiment as driven by the dataframe and config.
        metrics (Optional[SparkDataFrame]): Evaluation metric if user passes an evaluator.
    """

    train_timeframe: Dict
    test_timeframe: Dict
    run_name: Optional[str]
    result_df: SparkDataFrame
    metrics: Optional[SparkDataFrame]


@dataclass
class WalkForwardRunResults:
    """
    Class to capture results of walk forward model training and evaluation

    Attributes:
        run_results (List): A list of SingleMLRunResults.
        avg_metrics (Optional[Dict[str, float]]): The mean of the evaluation metrics across all validation folds.
        std_metrics (Optional[Dict[str, float]]): The standard deviation of the evaluation metrics across all validation
                                             folds.
    """

    run_results: List[SingleMLRunResults]
    avg_metrics: Optional[Dict[str, float]]
    std_metrics: Optional[Dict[str, float]]


class MLExperiment:
    """Generic experiment class for training, forecasting, and evaluating a model."""

    def __init__(self,
                 spark_session: SparkSession,
                 config: Dict,
                 model_cls: Type[BaseModel],
                 evaluator_obj: Optional[Union[BaseEvaluator, CompoundEvaluator]] = None):
        """
        ML Experiment class constructor.

        Args:
            spark_session (SparkSession): Current spark session object.
            config (Dict): Configuration dictionary.
            model_cls: Type of model we want to run the experiment with.
                       This is the class object that we instantiate within
                       the experiment with appropriate parameters
            evaluator_obj: Instance of BaseEvaluator class (e.g. WMapeEvaluator()) or CompoundEvaluator class used for
                           evaluation. If no evaluator is specified, evaluation will not be done.

        Raises:
            ValueError: When there is a mismatch between config specified algorithm and Model class passed
        """
        self.spark = spark_session
        self.config = config
        self.model_cls = model_cls

        # Validate config and model class align
        if self.config['model_params']['algorithm'] != self.model_cls.__name__:
            raise ValueError("Config & Model class object mismatch! Configuration specifies parameters for "
                             f"{self.config['model_params']['algorithm']} algorithm however experiment is passed "
                             f"a {self.model_cls.__name__} class object!")
        self.data_prep = DataPrepUtils(spark_session=self.spark, config_dict=self.config)
        self.evaluator = evaluator_obj

    def __str__(self):
        """
        Magic method to prints some model metadata when print() is called on a MLExperiment instance.

        Returns:
            return_str (str): Printout of some metadata values for the Model instance.
        """
        return_str = (
            "Experiment object for training, forecasting, and evaluation:\n"
            f"Model type: {self.model}\n"
            f"Evaluator: {self.evaluator}\n"
        )
        return return_str

    def set_evaluator(self, evaluator_obj: Optional[Union[BaseEvaluator, CompoundEvaluator]] = None):
        """
        Method to set the evaluator class of interest.

        Args:
            evaluator_obj (Optional[Union[BaseEvaluator, CompoundEvaluator]]): Instance of BaseEvaluator class
                (e.g. WMapeEvaluator()) or CompoundEvaluator class.
        """
        self.evaluator = evaluator_obj

    def walk_forward_model_training(self,
                                    df: SparkDataFrame,
                                    time_splits: Dict = None,
                                    verbose: bool = False) -> WalkForwardRunResults:
        """
        Method for one single pass of walk-forward cross-validation or model training on multiple splits of the data;
        ie. running the model through all walk-forward splits defined via parameters in the config and returning the
        mean and std_dev of the selected metric. This method uses multiprocessing.ThreadPool() to parallelize the
        multi-fold training across different threads.

        Args:
            df (SparkDataFrame): Dataset for model training, forecasting and evaluation
            time_splits (Dict): Dictionary of train-validation-holdout time splits. the format for the time splits
                                should look like so:
                                {
                                  "training": {
                                     "split1": {"start": "2020-01-01", "end": "2021-01-01"},
                                     "split2": {"start": "2020-01-01", "end": "2021-12-01"}
                                  },
                                  "validation": {
                                     "split1": {"start": "2021-01-01", "end": "2021-03-01"}
                                     "split2": {"start": "2020-12-01", "end": "2021-02-01"}
                                  }
                                  "holdout": {"start": "2021-03-01", "end": "2021-06-01"}
                                }
            verbose (bool): True/False flag for printing out runtime information for individual steps during
                            _single_train_test_eval().

        Returns:
            walk_forward_results (WalkForwardRunResults): Dataclass defined in this file that consists of a List of
                                                          SingleMLRunResults along with some optional parameters
                                                          such as mean & stdev of metrics
        """
        # Step 1: If time splits are provided as an argument here, use that
        # else use DataPrepUtils to get or create the splits
        if time_splits:
            # Given time splits is provided, we do some sanity validation checks on the
            # dataframe for null values, duplicate rows before progressing with further steps
            self.data_prep.validate_dataframe(df)
        else:
            # Time splits are created within tsff, based on dataframe and config specifications.
            # Sanity validation checks are done when these splits are created
            time_splits = self.data_prep.train_val_holdout_split(df)

        # Step 2: Validate format and data leakage in time splits dictionary and convert string dates to datetimes
        time_splits_dt = self.data_prep.validate_walk_forward_time_splits(time_splits)

        # Step 3: Run model training over different splits of data:
        num_splits = len(time_splits_dt['training'])
        print(f'Starting walk forward model training: Training and predicting over {num_splits} splits...')
        # Walk-forward model training via ThreadPool:
        with mp.pool.ThreadPool(mp.cpu_count()) as pool:
            run_results = pool.starmap(self.single_train_test_eval,
                                       [(df,
                                         time_splits_dt['training'][split_name],
                                         time_splits_dt['validation'][split_name],
                                         split_name,
                                         verbose) for split_name in time_splits_dt['training']])

        # Step 4: Capturing mean & stdev of metrics
        avg_metrics, std_metrics = None, None

        # If evaluator is provided, then compute and capture metrics
        if self.evaluator:
            metric_colnames = run_results[0].metrics.columns
            # Return metric values: mean & std
            avg_metrics = {c: mean([r.metrics.select(c).collect()[0][0] for r in run_results]) for c in metric_colnames}
            if len(run_results) == 1:
                std_metrics = {c: 0 for c in metric_colnames}
            else:
                std_metrics = {
                    c: stdev([r.metrics.select(c).collect()[0][0] for r in run_results]) for c in metric_colnames
                }

        # Step 5: Create output results dataclass
        walk_forward_results = WalkForwardRunResults(run_results=run_results,
                                                     avg_metrics=avg_metrics,
                                                     std_metrics=std_metrics)

        return walk_forward_results

    def single_train_test_eval(self,
                               df: SparkDataFrame,
                               train_timeframe: Dict,
                               test_timeframe: Dict,
                               run_name: str = "single_train_test_run",
                               verbose: bool = False) -> SingleMLRunResults:
        """
        Helper function for a single iteration (one fold) of a train-test-evaluation. Can be parallelized for use in
        multi-fold cross-validation.

        Args:
            df (SparkDataFrame): The dataframe to split on, featurize and fit a model.
            train_timeframe (Dict): The timeframe to train on. Format should look like this:
                                    {"start": 2018-01-01, "end": "2020-12-31"}
            test_timeframe (Dict): The timeframe to test on. Format should look like this:
                                   {"start": 2021-01-01, "end": "2021-12-31"}
            run_name (str): The name of the split (eg: 'split1') in a dictionary of train, test splits.
            verbose (bool): True/False flag for printing out runtime information for individual steps.

        Returns:
            result (SingleMLRunResults): A dataclass defined in this file that has the attributes:
                                         train_timeframe, test_timeframe, run_name, metrics, result_df.
        """
        t0 = time.time()
        # 1. Split dataset into train and val according to predefined range:
        df_train = self.data_prep.prepare_data_for_timeframe(df, train_timeframe)
        df_test = self.data_prep.prepare_data_for_timeframe(df, test_timeframe)
        if verbose:
            t1 = time.time()
            print(f'({run_name}) Finished splitting dataframe (time): {t1 - t0}')

        # 2. Featurize dataframe:
        features_utils = FeaturesUtils(self.spark, self.config)
        df_train_feats = features_utils.fit_transform(df_train)
        df_test_feats = features_utils.transform(df_test)
        if verbose:
            t2 = time.time()
            print(f'({run_name}) Finished featurization (time): {t2 - t1}')

        # 3. Instantiate model object for this run of training and predicting
        model = self.model_cls(model_params=self.config['model_params'], dataset_schema=self.config['dataset_schema'])

        # 4. Set feature columns:
        model.set_feature_columns(feature_colnames=features_utils.feature_colnames)

        # 5 & 6. Fit a model on train data and predict on validation data
        df_test_pred = model.fit_and_predict(df_train_feats, df_test_feats)
        if verbose:
            t3 = time.time()
            print(f'({run_name}) Finished fit and predict (time): {t3 - t2}')

        # 7. Compute evaluation metric if evaluator is provided (Optional)
        metrics = None
        if self.evaluator:
            metrics = self.evaluator.compute_metric_per_grain(
                df=df_test_pred,
                target_colname=model.target_colname,
                forecast_colname=model.forecast_colname,
                grain_colnames=[]
            )
            # Computing metrics
            if verbose:
                t4 = time.time()
                print(f'({run_name}) Finished metric computation (time): {t4 - t3}')

        # 8. Creating the output dataclass for run result
        result = SingleMLRunResults(train_timeframe=train_timeframe,
                                    test_timeframe=test_timeframe,
                                    run_name=run_name,
                                    result_df=df_test_pred,
                                    metrics=metrics)

        return result
