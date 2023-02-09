# Evaluation

This module includes implementations of common metrics relevant to time series forecasting. It also contains a function to save the forecast values to a specified database and table. The evaluators are setup in a hierarchical fashion as shown in the image below

![evaluators](../../docs/images/evaluators.png)

## base_evaluator.py

Implementation of the `BaseEvaluator` class. It defines the interface for all other evaluators and implements a method to compute the specific metric at any granularity level called `compute_metric_per_grain`.

## compound_evaluator.py

Implementation of the `CompoundEvaluator` class that can be used to manages multiple evaluators at once.

## wmape_evaluator.py

Implementation of the class WMapeEvaluator (Weighted Mean Absolute Percentage Error) and absolute accuracy measures. Inherits from `BaseEvaluator`.

## save_results.py

Function to save forecast results to specified database and table.
