import joblib
import pandas as pd
import numpy as np


class Predictor:
    def __init__(self):
        self.model = joblib.load('model/model.pkl')

    def predict(self, data: dict):
        """
        This method receives a new dataframe to make predictions with the
        regressor already trained
        :param data:
        :param df: cleaned dataframe ready to create predictions:
        :return: predictions for price
        """

        df_to_predict = pd.DataFrame(data, index=[0])
        df_to_predict = self.model.adjustToTrainedModelDF(df_to_predict)

        X = df_to_predict.drop(["price"], axis=1).to_numpy()

        return {"result": f"{np.exp(self.model.regressor.predict(X))}"}
