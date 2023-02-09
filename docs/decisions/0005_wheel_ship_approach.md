# Wheel distribution "ship" approach

## Context & Problem Statement

Time Series Forecasting Framework (TSFF) is a collection of classes, interfaces, and utilities used both in experimentation and for productionalizing models as part of the MLOps pipeline. Each of these interfaces can be leveraged at different stages of the data science lifecycle and have been setup to be extensible by users of the framework. A standard approach for easy portability of such a framework across multiple users and development environments is to create a built-package distribution format, also called a wheel. Wheel distribution is so ubiquitous that we don't consider that choice worthy of an ADR. Instead, the focus for this ADR is to outline how the wheel distribution will be shipped with each release, backward compatibility support and how we will do tagging and versioning for the wheel.

- For a deeper dive on the topic, here's a [post on python wheels](<https://realpython.com/python-wheels/>).

## Considered Options

Here are options for how we "ship" new versions of the package

- Manual installation of the wheel on clusters.
- Automated pipeline that installs the wheel on clusters.

## Decision drivers

- Customer security policies
- Ability to leverage new features quickly
- Minimize management burden

## Decision Outcome

There may be scenarios where a data scientist or developer may not have permissions to install libraries onto a dedicated cluster and has to rely on other stakeholders to manage this. This maybe due to a customer specific process or security policy. In these situations, every new version of the `tsff` python package will need to be manually installed on dedicated (all-purpose) clusters.

However, where there no issues related to installing packages on dedicated clusters, we do recommend this process be automated via MLOps pipelines with dedicated workstream representatives having required permissions.

For job clusters on the contrary, we have an MLOps CI/CD pattern in place that will package existing framework code into a wheel, create a build artifact and automatically install the artifact on clusters for training and scoring models.

In terms of versioning the wheel, we align with semantic versioning and explicit version support for a single production release as specified in [this ADR](0003-versioning-strategy.md) on versioning strategy. Given the broader code base will follow the Gitlab flow, maintaining short lived release branches for iterative framework development and tagging new updates to the wheel in a compatible manner makes management easier for a small team.
Further, maintenance of more than one production release is a heavy a burden for a small set of people continuously experimenting with new features and integrating to the package. Therefore, the team will support a single currently-deployed production release version of wheel at any one time.
