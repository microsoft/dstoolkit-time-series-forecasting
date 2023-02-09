# Troubleshooting guide

When using the experimentation framework, the table below illustrates typical issues:

|#| Issue? | Resolution |
|-|-|-|
| 1 | When executing the `run_ml_experiment_mlflow.py` notebook, the error illustrated below occurs. <br/> <br/> ![config_file_error](images/config_file_error.png) | Ensure the `config_filename` and `config_path` variables are assigned to the correct Json configuration file and path, respectively. <br/> <br/> ![corrected_config_path_model_training](images/corrected_config_path_model_training.png)|
| 2 | When executing the `create_config_random_forest.py` notebook, the error illustrated below occurs. <br/><br/> ![table_not_found](images/table_or_view_not_found.png) | 1. Ensure the configuration dictionary `dataset.db_name` and `dataset.table_name` is set correctly. <br/><br/> ![verify_config_dataset](images/verify_config_dataset_settings.png) <br/><br/> 2. Ensure the configuration dictionary is set to use a database and table that is accessible to the Azure Databricks cluster being used. <br/> <br/> ![verify_table_access](images/verify_access_to_tables.png)|

It is encouraged for users to work with TSFF code maintainers to collaborate on resolving any issues. Additionally, users are welcome to contribute to this troubleshooting guide to help accelerate resolutions for others to leverage.
