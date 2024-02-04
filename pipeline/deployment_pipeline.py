import numpy as np 
import pandas as pd 
from zenml import pipeline,step
from zenml.config import DockerSettings
# from materializer.custom_materializer import cs_materializer 
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step 
from zenml.steps import BaseParameters,Output

from steps.clean_data import clean_df
from steps.evaluate_model import model_eval
from steps.ingest_data import ingest_df
from steps.train_model import model_training



docker_settings = DockerSettings(required_integrations=[MLFLOW])


class DeploymentTriggerConfig(BaseParameters):
    min_accuracy:float = 0.92


@step 
def deployment_trigger(
    accuracy:float,
    config: DeploymentTriggerConfig
):
    '''Implements a simple model deployement trigger that looks at the input model accuracy and decides if it is good enough to deploy or not'''
    return accuracy >= config.min_accuracy




@pipeline(enable_cache=False,settings={"docker": docker_settings})
def continous_deployment_pipeline(
    data_path:str,
    min_accuracy:float = 0.92,
    workers:int =1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT):
    
    
    df = ingest_df(data_path=data_path)
    X_train,X_test,y_train,y_test = clean_df(df) 
    model = model_training(X_train,X_test,y_train,y_test)
    r2_score,rmse = model_eval(model,X_test,y_test) 
    deployment_decision = deployment_trigger(r2_score)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers = workers,
        timeout = timeout)
    