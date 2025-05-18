import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
from dataclasses import dataclass

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred
        
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
@dataclass   
class CustomData:
    carat: float
    depth: float
    table: float
    x: float
    y: float
    z: float
    cut: str
    color: str
    clarity: str

    def get_data_as_dataframe(self):
        try:
            custom_data_input = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x': [self.x],
                'y': [self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(custom_data_input)
            logging.info('Dataframe Gathered')
            return df
        
        except Exception as e:
            logging.info('Error occured during taking input data')
            raise CustomException(e,sys)

    



