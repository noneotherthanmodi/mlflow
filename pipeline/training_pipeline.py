import logging 
from zenml import pipeline

#importing in the sequence on which pipeline will work:
from steps.ingest_data import ingest_df 
from steps.clean_data import clean_df
from steps.train_model import model_training
from steps.evaluate_model import model_eval





@pipeline(enable_cache=True)
def pipeline_training(data_path: str):
    df = ingest_df(data_path)
    X_train,X_test,y_train,y_test = clean_df(df) 
    model = model_training(X_train,X_test,y_train,y_test)
    r2_score,rmse = model_eval(model,X_test,y_test)                   




