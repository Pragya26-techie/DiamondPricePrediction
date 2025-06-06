import sys
import os
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor

from src.utils import save_object
from src.utils import evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Splitting Dependent and Independent variable from train and test data")
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                'LinearRegression' : LinearRegression(),
                'Lasso' : Lasso(),
                'Ridge' :Ridge(),
                'Elasticnet':ElasticNet(),
                'DecisionTreeRegressor':DecisionTreeRegressor()
            }
            
            model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,models)
            print(model_report)
            print('\n============================================')
            logging.info(f'Model Report : {model_report}')

            ## To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'Best model Found, Model Name : {best_model_name},r2_score : {best_model_score}')
            print('\n=========================================')
            logging.info(f'Best model Found, Model Name : {best_model_name}, r2_score:{best_model_score}')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
        
                                                        
        except Exception as e:
            logging.info('Exception occured at model traning')
            raise CustomException(e,sys)
            
