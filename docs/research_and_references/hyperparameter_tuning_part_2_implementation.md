# Hyperparameter Tuning - Part 2: Exploration of Implementation Options

This is a continuation of [Hyperparameter Tuning - Part 1](hyperparameter_tuning_part_1_methods.md). In this doc we explore some implementation options for hyperparameter tuning. Mainly, we are interested in ways of parallelizing the tuning process such that we can take advantage of Spark's cluster of compute nodes in the Databricks environment.

Here is a brief overview of the options researched:

1. **Manual**: Manually perform the hyperparameter tuning via random search, in default Python.
    - This has the advantage of flexibility; we can set it up with more options and/or ease of usability.
    - However this is difficult to parallelize in Spark. In order to parallelize across different compute nodes, we'd have to rewrite our modeling, evaluation, and CV architecture in PySpark.
2. **Hyperopt** (more details below): Use the Hyperopt library to perform hyperparameter tuning.
    - Hyperopt is a library that has built-in features to parallelize across the Spark cluster. This should be (supposedly) much faster than single-threaded computations. Hyperopt also uses the [Tree-structured Parzen Estimator (TPE)](https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf) as its default search algorithm, which has been shown to outperform simple random search when used optimally.
    - The disadvantage, however, is that it is less flexible, and users will have to learn a few thing about the Hyperopt API in order to properly set up the search space.
    - AS OF MAY 2022: We are seeing unexpected behavior when using Hyperopt in Spark parallel mode.

## Hyperopt Library

[Hyperopt](http://hyperopt.github.io/hyperopt/) is a library for performing distributed asynchronous hyperparameter optimization. Parallelization with Spark clusters is supported natively. To perform hyperparameter tuning in Hyperopt generally involves 3 steps:

1. Set up an objective function that Hyperopt will work to minimize.
2. Define your hyperparameter search space using Hyperopt's [parameter expressions](http://hyperopt.github.io/hyperopt/getting-started/search_spaces/#parameter-expressions).
3. Run the Hyperopt optimization using `hyperopt.fmin()`.

The following is an example of the three steps described above:

```Python
# 0. Define datasets:
y_train, y_test = train_df['y'], test_df['y']
X_train, X_test = train_df.drop('y', axis = 1), test_df.drop('y', axis = 1)

# 1. Set up objective function (Random Forest example):
def hyp_obj(params):
    model = RandomForestRegressor(**params)
    model.fit(X = X_train, y = y_train)
    preds = model.predict(X_test)
    error = mean_squared_error(y_true = y_test, y_pred = preds)
    return {'status':STATUS_OK, 'loss':error}

# 2. Define hyperparameter search space using Hyperopt parameter expressions (several selected here as an example):
search_space = {
    "n_estimators": hp.choice("n_estimators", [50, 75]),
    "max_depth": hp.choice("max_depth", [20, 30, 40, 50]),
    "min_samples_split": hp.quniform("min_samples_split", 2, 8, 1),
    "bootstrap": hp.choice("bootstrap", [True, False])
}

# 3. Run tuning trials:
spark_trials = hyperopt.SparkTrials(parallelism = 8)
best = hyperopt.fmin(
    fn = hyp_obj,
    space = search_space,
    algo = hyperopt.tpe.suggest,
    max_evals = 32,
    trials = spark_trials
)
```

Overall, it is not difficult to set up. However, when defining the hyperparameter search space, users will have to understand the different Hyperopt "parameter expressions" which are used in the TPE algorithm to perform the optimization. These parameter expressions control the method by which each hyperparameter will be searched. For example, one should not confuse `hp.choice()`, which is used for selecting from *discrete categories* (where order doesn't matter) such as True/False, with `hp.quniform()`, which is used for selecting from *discrete ranges* (where order does matter) such as [3, 10]. In order to use Hyperopt properly, users must input both the search ranges and the method of search (the parameter expression) for each hyperparameter.

The complete list of parameter expressions is available [here](http://hyperopt.github.io/hyperopt/getting-started/search_spaces/#parameter-expressions)

## As of May 2022: Problems currently facing with Hyperopt

As of May 2022 we are encountering an issue with the parallelization capabilities of Hyperopt.

Description: The `SparkTrials()` class is used to trigger Hyperopt to run in in parallel and leverage the resources across a Spark cluster, for instance: `hyperopt.SparkTrials(parallelism = 8)`. Unfortunately, this parallelization is not behaving as expected.

- When comparing the single-threaded (without SparkTrials) version with the parallel version for an underlying Random Forest model, we see that the the parallel version's runtime is significantly longer than the single-threaded one.
- When comparing the single-threaded version with the parallel version for an underlying LightGBM model, we see a drastic speed-up for the parallel version that is way more of a runtime reduction than the parallelism parameter would suggest (ie: Parallelism was set to 8, but the speed up was 16x. We further tested this with Parallelism set to 1, and there still was a 16x improvement).

The `hyperopt_test.ipynb` notebook shows these examples in action.

## References

- **Hyperopt documentation:** [Hyperopt: Distributed Asynchronous Hyper-parameter Optimization](http://hyperopt.github.io/hyperopt/)

- **Bergstra paper on Bayesian hyperparameter optimization:** Bergstra, et al. [Algorithms for Hyper-Parameter Optimization](https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf)

- **Databricks tutorial on Hyperopt:** [How (Not) to Tune Your Model With Hyperopt](https://databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html)
