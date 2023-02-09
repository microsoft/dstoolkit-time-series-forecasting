---
# These are optional elements. Feel free to remove any of them.
status: proposed (but implemented)
date: 2022-08-05
---
# Implement Lags Feature Using applyInPandas Without Imputation

## Context and Problem Statement

Time series data is often featurized using "lags" to allow future predictions to be based on previous periods (e.g. tomorrow will be similar to today). The Experimentation Framework should contain a Lags Featurizer that produces these features in a consistent fashion so that all experimentation streams can use it to get lag features in a consistent and performant manner. How should it be implemented, and how does it deal with gaps in the series?

**Note: This is a retrospective ADR, documenting existing code in an effort to understand it and present alternative options in the event it needs to be reconsidered.**

<!-- This is an optional element. Feel free to remove. -->
## Decision Drivers

- Featurizer should deal with univariate and multivariate series
- Should support ability to "lag" multiple columns, but possibly a subset of all (e.g. don't lag `product_name`)
- Should support lagging at multiple intervals in a single call (e.g. lag by 1, 2, and 7 days)
- Should be able to leverage the featurizer for time-series data specified at different granularities
- Featurizer should perform reasonably for small # of time series (thousands) but scale to perform reasonably for millions. Reasonably means experimenters should be able to do multiple experiments per day on a dedicated cluster.

## Considered Options

- Spark DataFrame using applyInPandas and Pandas's `DataFrame.shift` method
- Spark DataFrame using `pyspark.sql.functions.lag`
- Pandas DataFrame using Pandas's `DataFrame.shift` method

## Decision Outcome

Chosen option: Spark DataFrame using applyInPandas and Pandas's `DataFrame.shift` method, because `pyspark.sql.functions.lag` will only lag a single column at a time and is not time-aware, and using a Pandas dataframe from the start (i.e. not `applyInPandas`) will not scale.

<!-- This is an optional element. Feel free to remove. -->
### Positive Consequences

- Scales reasonably well

<!-- This is an optional element. Feel free to remove. -->
### Negative Consequences

- Can struggle on small series over native Pandas as the data fan-out/fan-in and query-plan optimization can be prohibitive

<!-- This is an optional element. Feel free to remove. -->
## More Information

There is additional implementation here regarding the forecast horizon and padding dataframe, which I'm not sure I understand the reasoning behind. This will be documented in a separate ADR as it has impact on other featurizers as well.
