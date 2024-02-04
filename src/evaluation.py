import logging
from abc import ABC, abstractmethod
import numpy as np 
from sklearn.metrics import mean_squared_error,r2_score

class Evaluation(ABC):

    '''calculates the score of the model...'''
    @abstractmethod
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        pass


class MSE(Evaluation):
    '''calculates the mean squared error'''

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('calculating MSE...')
            mse = mean_squared_error(y_true,y_pred)
            logging.info('MSE: {}'.format(mse))
            return mse 

        except Exception as e:
            logging.error(f'Error in calculating MSE {e}')
            raise e 


class R2(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('calculating R2 scores...')
            r2 = r2_score(y_true,y_pred)
            logging.info(f'R2_score: {r2}')
            return r2 
        except Exception as e:
            logging.error(f'Error in calculating R2 {e}')
            raise e 
        

class RMSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating rmse scores...')
            rmse = mean_squared_error(y_true,y_pred)
            logging.info(f'rmse score: {rmse}')
            return rmse
        
        except Exception as e:
            logging.error(f'Error in calculating rmse {e}')
            raise e 