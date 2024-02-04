import logging 
from abc import ABC,abstractmethod
from typing import Union

import pandas as pd 
import numpy as np
from pandas.core.api import Series as Series 
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):

    "abstract class for defining handling of data."
    @abstractmethod
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass 


class DataPreProcessingStrategy(DataStrategy):

    "Strategy for preprocessing of data."

    def handle_data(self, data:pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.drop(
                [
                    'order_approved_at',
                    'order_delivered_carrier_date',
                    'order_delivered_customer_date',
                    'order_estimated_delivery_date'
                ],
                axis=1)
            
            data['product_weight_g'].fillna(data['product_weight_g'].median(),inplace=True)
            data['product_length_cm'].fillna(data['product_length_cm'].median(),inplace=True)
            data['product_height_cm'].fillna(data['product_height_cm'].median(),inplace=True)
            data['product_width_cm'].fillna(data['product_width_cm'].median(),inplace=True)
            data['review_comment_message'].fillna('No Review',inplace=True)

            data = data.select_dtypes(include=[np.number])
            col_to_drop = ['order_item_id','customer_zip_code_prefix']
            data = data.drop(columns=col_to_drop,axis=1)
            return data

        except Exception as e:
            logging.error(f"Error cleaning the data {e}.")
            raise e 
        

class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop(['review_score'],axis=1)
            y = data['review_score']
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
            return X_train,X_test,y_train,y_test        #returning attributes: y_train and test is pd.Series and X_train and test is pd.DataFrame
        except Exception as e: 
            logging.error("Error training the data {}".format(e))
            raise e 
        

class DataCleaning:
    def __init__(self,data:pd.DataFrame, strategy:DataStrategy):
        self.data = data 
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame,pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error('Error in handling the data {e}'.format(e))
            raise e
        
