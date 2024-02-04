import logging 
import pandas as pd 

from zenml import step 

from sklearn.base import RegressorMixin
from src.evaluation import MSE,R2,RMSE

from typing import Tuple
from typing_extensions import Annotated 


@step  
def model_eval(model:RegressorMixin,
    X_test:pd.DataFrame,
    y_test:pd.DataFrame) -> Tuple[
        Annotated[float,'r2_score'],
        Annotated[float,'rmse']
    ] :

    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test,prediction)

        r2_class = R2()
        r2_score = r2_class.calculate_scores(y_test,prediction)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test,prediction)

        return r2_score,rmse 

    except Exception as e:
        logging.error(f'error at model_eval: {e}')
        raise e 

