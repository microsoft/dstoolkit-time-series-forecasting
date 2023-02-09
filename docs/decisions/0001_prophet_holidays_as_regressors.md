# Manage Prophet holidays as regressors

## Context and Problem Statement

Prophet is a package which forecasts time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

There are different ways to inform Prophet about the existing holidays, and we need to decide the most appropriate method for our needs.

## Decision Drivers

- Accuracy
- Run time
- Flexibility (e.g. adding a weight to each holiday or day)
- Generalizable

## Considered Options

- Constructor parameter `holidays`
- Method `add_regressor`
- Method `add_country_holidays`

## Decision Outcome

Chosen option: "Method `add_regressor`", because it achieves a similar accuracy with a similar run time, provides more flexibility and generalizes better to other algorithms/packages.

### Flexibility

In the following posts it is stated that "Constructor parameter `holidays`" is internally translated into a binary implementation of "Method `add_regressor`". This makes both solutions equivalent in terms of accuracy and runtime, but means that the `add_regressor` method is more flexible.

- [holiday vs regressor](https://github.com/facebook/prophet/issues/1651)
- [The difference between using 'holidays' and 'add_regressor' when identifying a specific holiday](https://github.com/facebook/prophet/issues/1491)

Option "Method `add_country_holidays`" is quite inflexible, as it relies on the package built-in collection of country-specific holidays. It would be useful if the list of holidays (or other events) were not available.

### Generalizable

Assuming ARIMA is to be implemented using the Python package `pmdarima`, the "Method `add_regressor`" better matches how holidays are managed in ARIMA.

As described in the documentation linked below, ARIMA handles holidays through a matrix with as many rows as observations in the TS. This matrix is exactly the same structure as the columns added as regressors in Prophet.

- [API Reference: pmdarima.arima.auto_arima](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html)

## More Information

In the following link there is a description and examples of how to use the different parameters and methods discussed in this document:

- [Seasonality, Holiday Effects, And Regressors](https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html)
