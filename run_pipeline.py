from pipeline.training_pipeline import pipeline_training
from zenml.client import Client



if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    pipeline_training(data_path = "D:\mlOps\dataset\olist_customers_dataset.csv")



# mlflow ui --backend-store-uri file:C:\Users\uditm\AppData\Roaming\zenml\local_stores\67582014-01db-4e40-8b68-1346db0156d8\mlruns





#AFTER BRINGING MLFLOW TRACKER IN THE PROJECT:  
    
# The project can only be executed with a ZenML stack that has an MLflow experiment tracker 
# and model deployer as a component. Configuring a new stack with the two components are as follows:

# zenml integration install mlflow -y
# zenml experiment-tracker register mlflow_tracker --flavor=mlflow
# zenml model-deployer register mlflow --flavor=mlflow
# zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set