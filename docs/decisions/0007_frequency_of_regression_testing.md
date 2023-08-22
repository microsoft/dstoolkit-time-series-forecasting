---
status: proposed
---
# Time Series Forecasting Accelerator frequency of regression testing

## Context and Problem Statement

The Time Series Forecasting Accelerator (TSFA) is a collection of classes, "interfaces", and utilities used to train as well as evaluate models. TSFA will support and enable multiple workstreams as they leverage its packaged utilities for featurization, modeling and operationalization. Use of the package and newer business use-cases will drive improvements and enhancements, new features, models and utilities back to TSFA. Due to this we need a way to periodically conduct end-to-end testing (regression testing) to ensure we have a stable codebase that can be used by Experimentation Workstreams as bugs are fixed, new functionality is introduced, and existing functionality is enhanced. At the same time, we need to ensure end-to-end testing does not impact the experimentation flow and potentially limit/delay plans for production deployments.

## Decision Drivers

- Since user Data Scientists can contribute to the codebase, we want the ability to regularly test changes such as bug fixes, new functionalities, and enhancements to existing functionalities in order to ensure the codebase works as expected.
- Conduct end-to-end test periodically enough to ensure bugs are identified sooner.
- Identify and resolve any bugs before MLOps integration for a planned production release.
- Conduct end-to-end testing on different sized datasets (such as small, medium, and large test datasets) to ensure execution times are tracked as the codebase evolves.
- Update or define a new set of test datasets as new functionality is introduced to the codebase.
- Ability to track end-to-end execution time to train a model as the codebase evolves.
- Automation of the end-to-end tests would enable periodic job scheduling.
- Need to schedule end-to-end testing at a time that would not impact Experimentation Workstreams cluster usage.

## Considered Options

- End-to-end testing on every commit
- End-to-end testing on every major change
- Weekly end-to-end testing on codebase in main branch
- End of Sprint end-to-end testing on codebase in main branch
- Every major change would require end-to-end test on small dataset, end of sprint end-to-end testing on codebase in main branch using test datasets

## Decision Outcome

**Chosen option:** Every major change would require end-to-end test on small dataset, end of sprint end-to-end testing on codebase in main branch using test datasets.

- On every sprint, any bugs in `main` branch can be identified, and any concerns on execution time would be identified.
- EF Champions will be aware of the expected execution times based on changes committed earlier in the sprint.
- However, when the test datasets need to be updated, EF Champions will require bandwidth to do the work. Additionally, if the regression testing process is not automated, EF Champions will require bandwidth per sprint to conduct the tests.
- Additionally, on every major change commit, any integration issues are identified early.

## Pros and Cons of the Options

For all options considered:

- The regression tests should be executed by EF Champions or Experimentation Workstreams that are accountable for the changes being committed. However, the results of the regression tests such as execution times for each test dataset should be communicated to all EF Champions for awareness.
- Once changes are merged (using existing PR process with peer code reviews, linting checks and unit testing), the `tsfa` codebase version should be tagged in AzDO.
- If the changes to the codebase after the last regression test requires an update to the test datasets, this needs to be completed prior to the next regression test. Updates could include a data refresh, and/or new schemas for the datasets to ensure new featurizers can be computed appropriately.
- Regression test results such as execution time will be logged to MLFlow experiment as a metric.
- The framework codebase version (`tsfa-vx.x.x`) should be logged to MLFlow for tracking.

### End-to-end testing on every commit

- On every commit, regression tests comprising of small, medium, and large sized test datasets should be conducted.
- Good, because:
  - Any impact on metrics like execution time is identified early when changes are committed to `main` branch.
- Bad, because:
  - There could be many commits to `main` branch within a day. To conduct regression testing considering small, medium and large sized test datasets will take time to execute and expensive on a daily basis.
  - If an update on the small, medium, and large sized test datasets is required, additional effort is required to update these datasets. There is a possibility that during the dataset update, further changes are committed.

### End-to-end testing on every major change

- Major changes include enhancements such as
  - adding new parameters to configuration file for a new featurizer or model class,
  - any refinements to configuration file that lead to changes in the signatures of the classes, class methods, and module functions,
  - adding a new featurizer or model class,
  - refinements to save forecast results data table schema,
  - refactoring data loading functionality,
  - refinements on existing featurizers or model classes (including featurizer and model orchestrator classes)
- On any major change, regression tests comprising of small, medium, and large sized test datasets should be conducted.
- Good, because:
  - Major changes include changes to configuration file schemas, adding new featurizer/model classes, enhancements to featurizer/model orchestrators, data loading, and saving forecast results to data tables. These cover core framework functionality, and any issues due to these changes would be identified.
  - Any of these major changes that have a negative impact on metrics such as execution time will be identified.
  - Reduce compute costs as regression tests are only conducted on major changes.
- Bad, because:
  - Minor changes that could have a negative impact on metrics such as execution time may not be detected, e.g. a new unit test may take long to execute during the PR process.
  - Bug fixes that could also have a negative impact on metrics such as execution time may not be detected.
  - Certain performance metrics such as execution time are also impacted by Azure Databricks cluster configurations. When conducting regression tests only major changes, the changes to these configurations may not be detected in a timely fashion.
  - To measure metrics such as execution time, clusters that have many notebooks attached should not be used as this can impact the processing time of other Data Scientist users jobs. Identifying a cluster to use for regression testing can be challenging on a weekly basis.

### Weekly end-to-end testing on codebase in main branch

- Once a week, regression tests are conducted on the `tsfa` codebase in `main` branch to ensure code executes as expected and metrics such as execution time are similar to previous weeks or tests.
- The motivation for this option is to provide a trade-off between early discovery of performance degradation or breaking changes without slowing down the day-to-day cadence of work or putting undue pressure on experimentation clusters.
- Good, because:
  - Any impact on metrics like execution time is identified on a weekly basis. This will enable faster identification of changes that resulted in the performance change, i.e. finite number of changes within the week that resulted in the performance change.
- Bad, because:
  - To measure metrics such as execution time, clusters that have many notebooks attached should not be used as this can impact the processing time of other Data Scientist users jobs. Identifying a cluster to use for regression testing can be challenging on a weekly basis.
  - If an update on the test datasets is required, additional effort is required to update these datasets. This could be challenging on a weekly basis.
- Weekly regression testing may be too frequent as usual release cycles and/or sprints are typically longer than 1 week. Due to this, major changes may not be ready and merged to `main` branch within a week.

### End of Sprint end-to-end testing on codebase in main branch

- At the end of a sprint, regression tests are conducted on the `tsfa` codebase in `main` branch to ensure code executes as expected and metrics such as execution time are similar to previous regression tests.
- The reasoning behind this option is to provide discovery of performance degradation or breaking changes before attempting any end-of-sprint pre-release efforts, while avoiding slowing down the day-to-day cadence of work, putting undue pressure on experimentation clusters, or unduly impacting the work of the Experimentation Workstream team during the core of development time on the sprint.
- Good, because:
  - During sprint planning, the work can be planned appropriately allocating bandwidth to EF Champion(s) to conduct regression tests.
  - At the end of every sprint, EF Champions (and workstreams that they represent) will know the expected performance (such as execution time) of `tsfa` codebase.
  - If there are any issues in performance due to changes, the changes can be identified quickly as there finite number of changes per sprint.
  - If there are any issues in performance and this needs to be addressed, the work can be planned for the next sprint.
- Bad, because:
  - Cluster availability (i.e. identify cluster with minimal usage) can be challenging at the end of a sprint.
  - If there are changes that require an update on the test datasets prior to the end of the sprint, adequate planning is required to ensure sufficient capacity is available to do this work.

### Every major change would require end-to-end test on small dataset, end of sprint end-to-end testing on codebase in main branch using test datasets

- The same major enhancements illustrated in the "end-to-end testing on every major change" option apply. However, when these major enhancements are committed  to `main` branch, the codebase is tested end-to-end using a small dataset.
- Additionally, similar to the "end of sprint end-to-end testing on codebase in main branch option, the `tsfa` codebase in `main` branch is regression tested at the end of each sprint to ensure metrics such as execution time are similar to previous regression tests.
- The reasoning behind this option is to identify potential integration issues early, when the new major changes are committed. Same reasons as previously stated for testing end-to-end at the end of each sprint also apply to this option
- Good, because:
  - Can allocate capacity for regression testing in each sprint, and correction of bugs discovered for the next sprint.
  - Integration problems are identified early on each major change commit.
  - At the end of every sprint, EF Champions (and workstreams that they represent) will know the expected performance (such as execution time) of `tsfa` codebase.
- Bad, because:
  - Cluster availability may be limited and challenging to plan at the end of a sprint.
  - Adequate planning is required to ensure capacity to update (if required) test datasets prior to end of the sprint.
  - Adequate planning is also required to ensure bandwidth to update (if required) the small dataset used for end-to-end testing on major changes.
