import joblib
import pandas as pd
import numpy as np

from model.Model import Model
from preprocessing.cleaning_data import DataProcessor


class Predictor:
    def __init__(self):
        self.model = Model()
        self.dataProcessor = joblib.load('preprocessing/DataProcessor.pkl')

    def predict(self, data: dict):
        """
        This method receives a new dataframe to make predictions with the
        regressor already trained
        :param data:
        :param df: cleaned dataframe ready to create predictions:
        :return: predictions for price
        """
        self.dataProcessor.checkErrors(data)
        df_to_predict = pd.DataFrame(data, index=[0])
        df_processed_data = self.dataProcessor.preprocess(df_to_predict)
        df_to_predict = self.model.adjustToTrainedModelDF(df_processed_data)

        X = df_to_predict.drop(["price"], axis=1).to_numpy()

        return {"result": f"{np.exp(self.model.regressor.predict(X))}"}
