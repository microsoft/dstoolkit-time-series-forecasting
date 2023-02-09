# How to add a new model to Time Series Forecasting Framework?

In this document, we give a brief overview of how models are setup in tsff, and how a developer can add a new one.

![models](./images/model_classes.png)

Models have been structured in a hierarchical fashion as shown in the image above and it depicts:

- `BaseModel` is structured as an abstract class from which all other models inherit.
- We have two wrapper classes
  - `UnivariateModelWrapper` for univariate models and
  - `MultivariateModelWrapper` for multivariate models
  - Both the classes inherit from `BaseModel`.
  - Additionally, the `MLLibRegressorWrapper` class inherits from `MultivariateModelWrapper` to provide more Pyspark ML models for experimentation.
  - The `RandomForestRegressor` is a child class of `MLLibRegressorWrapper`
- To add a new model, the user data scientist or the developer can create a new model class, inheriting from one of these wrappers if appropriate, else inherit from `BaseModel`. Within this class, specific implementations for fit and predict methods can be added.
- Each model can be specified as an ML algorithm in the configuration file, together with the required hyperparameters.

Below is an example showing a step by step method for the addition of a new model

## Step-1: Add the model package<br>

This step will enable the application of the algorithm needed for uni/multi-variate model<br>

- Add the package and the dependencies needed for the model in the requirements file `.azure_pipelines/requirements-pr.txt`
<br>The figure below illustrates the addition of `prophet package` <br><br>
![prophet](./images/prophet_package.png)
<br>This will ensure that you have the package installed in the library<br>

## Step-2 : Create a new python script with the model name<br>

- Create a new python script for the new model as `[model_name].py` in the folder `tsff/models`
 <br>The figure below illustrates the model prophet.py added to the folder</br><br>
 ![prophet_file](./images/model_name_prophet.png)
<br>

## Step-3:  Import the package libraries and the model wrapper

- Import the necessary package libraries needed for the model in the new file
- Also, import the model wrapper<br>
  - If the model is univariate import the wrapper as follows:<br>
   `from tsff.models import UnivariateModelWrapper`<br>
  - If the model is multivariate import the wrapper as follows:<br>
   `from tsff.models import MultivariateModelWrapper`<br>
  - If the model would not need these wrapper classes, import the `BaseModel`<br>
   `from tsff.models.base_model import BaseModel`

## Step-4:  Create class and the functions

- Create a class for the model object
<br>Below is an example snippet showing the model class
<br>Prophet- Univariate<br>
![prophet_class](./images/prophet_class.png)<br>
<br>RandomForest-Multivariate<br>
![randomforest_class](./images/randomforest_class.png)
- Create function to fit and predict<br>
  - If the model is univariate, create the function- `_fit_and_predict_single_ts_wrapper`<br>
 ![univariate_fit_pred_wrapper](./images/univariate_fit_predict_wrapper.png)<br>
 This function captures the class instance attributes used in the inner function<br>
 Within this function, create the UDF(user defined function) with the specific implementation of the new model fit and predict methods for a single time series for the univariate model<br>
<br>
  - If the function is multivariate,write the implementation of the individual fit and predict functions of the new model aligned to the multivariatewrapper class.<br>
  ![univariate_fit_pred_wrapper](./images/multivariate_fit_predict.PNG)<br>
- The output result of Univariate model should return a pandas dataframe while multivariate should return a SparkDataFrame
- Ensure the output dataframe contains all the columns along with the forecast column

## Step–5 Import the model class in the init file

- As a next step, import the newly defined model class into the `__init__.py` file in the models folder and also add the class in the `__all__` variable list
<br><br>
![model_init](./images/add_model_init.png)
<br>

## Step-6 Add the model and hyperparameters in the Config dictionary

- Once the new model is added, define the algorithm and hyperparameters in the `model_params` block in the config dictionary.<br>
Below is an example showing the model_params in the config for Prophet<br>
 ![Prophet_params](./images/prophet_params.png)

## Step-7 Add unit tests for the new model

- Write unit test cases for the new model and add it in the path `tsff/test/models`
- Write the script to test independent functionalities written in the new model in the path `tsff/notebooks/module_tests`.

## Step-8 Add Sample Notebook for the model

- Add the sample notebook that illustrates the configuration dictionary schema for the model as  `run_01_create_config_[model_name]` in the path `tsff/notebooks/samples`.<br>
This notebook will create a parameterized Json configuration file that specifies the experiment parameters.

## Step-9 Raise the PR in Azure DevOps

- Once the new model contribution development is completed, tested and validated to ensure the model class works as expected, raise a PR and add the `tsff` team as required reviewers.
