# ADR 006: Suggestions for improving the extensibility of tsff

- status: {proposed}
- date: {2022-09-22 when the decision was last updated}
- deciders: EF core team
- consulted: EF champions
- informed: All DS on FP&A

## Context and Problem Statement

The `tsff` package is the standardized and agreed upon framework for modeling and experimentation in the FP&A project.
Data scientists are leveraging the framework for the specific problem they work on.
However, in version 2.0, an implementation of a new functionality (e.g. model, evaluation, feature engineering)
requires the code to be merged into the framework, and adaptations to the core functionality.

Here are two examples:

1. Adding a new model requires a change to a hard-coded list of implemented models in model.py

    Different streams might require very specific logic to be implemented in models,
    which might not be applicable to other cases.
    One example is post-processing the model results to remove all positive predictions (in case of COGS).
    While this could be achieved as an external post-process phase,
    it would be cleaner to have it as part of the modeling logic.

2. Customizing the evaluation logic requires an api change in the framework

    While the framework has a `wmape` function, users have no control over the weights applied,
    and have no way of controlling how to compute or apply weights without changing framework code.

### Extension capabilities available with other frameworks

Frameworks such as `scikit-learn` and `sktime` use base classes which define an interface that is then used
across the framework. In Scikit-learn there is a [BaseEstimator class](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator)
and base classes for the different model types (e.g., classification, regression).

In sk-time there's a similar approach with a [BaseEstimator class](https://www.sktime.org/en/stable/api_reference/auto_generated/sktime.base.BaseEstimator.html#sktime.base.BaseEstimator).

This is also the typical architecture in PyTorch, Tensorflow and many other frameworks.

The main advantage is that the model implementation is decoupled from the implementation of auxiliary methods,
such as evaluation, cross-validation, experiment tracking and more.
These capabilities are in the core framework, and can be utilized with a model that is developed outside the framework,
as all models have an agreed upon interface.

The same architecture applies to evaluation metrics (for example, sktime has a [`BaseMetric` class](https://github.com/alan-turing-institute/sktime/blob/bd8d7c531d2dc3eb947ae31613dd929dc9f0d72f/sktime/performance_metrics/base/_base.py#L11))

### Proposed change

1. Create base classes for models and evaluators, or reuse those in sklearn/sktime
2. Decouple the experimentation logic from the modeling logic in `model.py`:
    1. Have the `__init__` function accept model and evaluator instantiated objects into the experimentation flow.
    2. Remove the need to instantiate a model inside the Model class, and to change code to accept new models.
3. Rename `Model` to `MLExperiment` as to clarify the decoupling of modeling logic with experimentation logic.

Example flow:

```python
class BaseModel(ABC):
    #init, fit, predict, set_feature_columns etc.

class MyCoolModel(BaseModel):
    # Implement my logic into fit/predict and define my model's parameters


class BaseEvaluator(ABC):
    # init, evaluate method

class WMapeEvaluator(BaseEvaluator):
    # Implementation of wmape

class BrandWeightedMapeEvaluator(WmapeEvaluator):
    # Logic on how to apply brand importance as weights during evaluation

class Featurizer(ABC):
    # fit, fit_transform, transform

class MyFeaturizer(Featurizer):
    # Specific featurization logic


class MLExperiment: #class is called Model in tsff
    def __init__(self, model_cls:Type[BaseModel], 
                       evaluator:BaseEvaluator, 
                       featurizer:Optional[Featurizer], ...):
        # store the model for reuse during cross validation
        self.model_cls = model_cls
        self.evaluator = evaluator
        self.featurizer = featurizer

    def _single_train_test_eval(self):
        # run one experiment

        # create features
        if self.featurizer:
            ...

        # train model
        model = self.instantiate_model()
        model.fit(...)

        # Predict
        preds = model.predict(...)

        # Evaluate
        results = self.evaluator.evaluate(preds, actuals, ...)
        
        ...
```

Note that in this example, the featurizer is part of the MLExperiment logic. It could also be part of the modeling logic.

### Positive Consequences

- Users would be able to customize the framework to their needs without a requirement to go through framework code.
- The same approaches could be extended to other parts of the framework,
such as the cross-validation strategy, which would allow even more customizability.

### Negative Consequences

- This would require an API change to the framework, but by adapting the existing configuration logic, this change could be minimal.
- Evaluators must be stateless as classes are instantiated outside the execution flow.
If for any reason there is leakage between folds, the user might not know about it.
- As this change would allow anyone to create models independently, we might lose some of the alignment on common code and methods. A potential mitigation for this is continuous alignment through EF champions.
