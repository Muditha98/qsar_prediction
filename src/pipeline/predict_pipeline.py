import os,sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)




class CustomData:
    def __init__(  self,
        CIC0: float, 
        SM1_DzZ: float, 
        GATS1i: float, 
        NdsCH: int,  
        NdssC: int, 
        MLOGP: float
        ):

        self.CIC0 = CIC0

        self.SM1_DzZ = SM1_DzZ

        self.GATS1i = GATS1i

        self.NdsCH = NdsCH

        self.NdssC = NdssC

        self.MLOGP = MLOGP


#'CIC0', 'SM1_DzZ', 'GATS1i', 'NdsCH',  'NdssC', 'MLOGP'

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "CIC0": [self.CIC0],
                "SM1_DzZ": [self.SM1_DzZ],
                "GATS1i": [self.GATS1i],
                "NdsCH": [self.NdsCH],
                "NdssC": [self.NdssC],
                "MLOGP": [self.MLOGP]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)