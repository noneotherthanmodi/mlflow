import logging 
from zenml import step 
import pandas as pd
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

import mlflow
from zenml.client import Client 
# import sklearn.autolog

experiment_tracker = Client().active_stack.experiment_tracker



@step (experiment_tracker=experiment_tracker.name)
def model_training(
    X_train:pd.DataFrame,
    X_test:pd.DataFrame,
    y_train:pd.Series,
    y_test:pd.Series,
    config:ModelNameConfig) -> RegressorMixin:
    

    try:
        model = None
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train,y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error(f'Error in training Model {e}.')
        raise e 


