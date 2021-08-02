import json
import re

import joblib
import pandas as pd
import numpy as np

from exceptions.InvalidExpression import InvalidExpression


class DataProcessor:

    def __init__(self):
        self.unique_categorical_values = dict()

    def preprocess(self, df: pd.DataFrame, isTrainingSet: bool = False) -> pd.DataFrame:
        """
        This method will clean the dataframe deleting duplicated rows,
        fixing errors, eliminating outliers. If visualize_flag set
        will plot correlation map and relationships between
         the variables and the target price
        :param isTrainingSet:
        :param df: dataframe to clean
        :return: cleaned dataframe
        """

        # We clean first all the entirely empty rows
        df.dropna(how="all", inplace=True)

        # We delete the blank spaces at the beginning and end of each string
        df.apply(lambda x: x.strip() if type(x) == str else x)

        # Fixing errors
        # Fixing variable hasFullyEquippedKitchen
        if "kitchenType" in df.columns:
            has_hyperEquipped = df["kitchenType"].apply(
                lambda x: 1 if x == "HYPER_EQUIPPED" else 0
            )
            has_USHyperEquipped = df["kitchenType"].apply(
                lambda x: 1 if x == "USA_HYPER_EQUIPPED" else 0
            )
            df.hasFullyEquippedKitchen = (
                    has_hyperEquipped | has_USHyperEquipped
            )

        if 'price' in df.columns:
            # Dropping rows with price as NaN values
            df = df[df["price"].notna()]
            df = df[df["area"].notna()]
            df = df[df["buildingCondition"].notna()]

            # Dropping duplicated values
            df = df.drop_duplicates(
                subset=["area", "price"], keep="last"
            )

            # Dropping outliers
            df = df[df["price"] < 6000000]
            df = df[df["area"] < 1350]

        if isTrainingSet:
            # cleaning features with less than 5 occurrences
            features = [
                "postalCode",
                "facadeCount",
                "subtypeProperty",
                "bedroomsCount",
            ]

            if df.shape[0] > 1000:
                for feature in features:
                    df = df[
                        df[feature].map(df[feature].value_counts()) > 5
                        ]

            for feature in ['postalCode', 'buildingCondition', 'subtypeProperty']:
                self.unique_categorical_values[feature] = df[feature].unique()

            joblib.dump(self, 'preprocessing/DataProcessor.pkl')

        # Creating new variable adding outside surface
        df.terraceSurface.fillna(0, inplace=True)
        df.gardenSurface.fillna(0, inplace=True)
        df["outsideSpace"] = (
                df["terraceSurface"] + df["gardenSurface"]
        )

        # Filling nan values to 0 for facadeCount
        df.facadeCount.fillna(0, inplace=True)

        # Deleting least correlated, reformatted and constant columns
        to_drop = ['kitchenType', 'typeProperty', 'typeSale', 'subtypeSale', 'isFurnished', 'terraceSurface', 'gardenSurface']

        for feature in to_drop:
            if feature in df.columns.to_list():
                del df[feature]


        # List of categorical values
        categorical_features = [
            "postalCode",
            "buildingCondition",
            "subtypeProperty",
            "hasFireplace",
            "hasSwimmingPool",
            "hasGarden",
            "hasTerrace",
            "hasFullyEquippedKitchen",
        ]

        # Transform categorical variables into features
        for feature in categorical_features:
            cv_dummies = pd.get_dummies(df[feature])
            if feature.startswith('has'):
                if cv_dummies.columns.__len__() == 1:
                    if cv_dummies.iloc[0, 0] == 0:
                        cv_dummies.columns = [feature + "False"]
                    else:
                        cv_dummies.columns = [feature + "True"]
                else:
                    cv_dummies.columns = [feature + "False", feature + "True"]
            df = pd.concat([df, cv_dummies], axis=1)
            del df[feature]

        df = self.rescale(df)

        return df.reset_index(drop=True)

    @staticmethod
    def rescale(df):
        """This static method will standardize some of the features,
        all the areas (square meter) will be rescaled into their square root
        and the price into its logarithm
        :param df: cleaned data frame to resale
        :return: rescaleded dataframe
        """

        if 'price' in df.columns.to_list():
            df["price"] = np.log(df["price"])

        df["area"] = np.sqrt(df["area"])
        df["outsideSpace"] = np.sqrt(df["outsideSpace"])
        df["landSurface"] = np.sqrt(df["landSurface"])

        return df

    def checkErrors(self, data: dict):
        columns = ['postalCode', 'subtypeProperty', 'bedroomsCount', 'area', 'hasFullyEquippedKitchen',
                   'hasFireplace', 'hasTerrace', 'terraceSurface', 'hasGarden', 'gardenSurface', 'landSurface',
                   'facadeCount', 'hasSwimmingPool', 'buildingCondition']

        missing = ''

        # First check if there is any value missing
        for column in columns:
            if data is None:
                raise InvalidExpression("No json object sent")
            elif column in data:
                # Checking error on postalCode field
                if column == 'postalCode':
                    if re.match(r'^\d{4}$', str(data[column])):
                        if 0 > data[column] or data[column] > 9999:
                            raise InvalidExpression("The zip code should be a number between 0 and 9999.")
                        else:
                            if data[column] not in self.unique_categorical_values[column]:
                                raise InvalidExpression("Sorry this code is not into our model. We are working to add"
                                                        "more data to it.")
                    else:
                        raise InvalidExpression("The zip code should be expressed as an integer")
                ## Checking error on bedroomsCount field
                elif column in ['bedroomsCount', 'facadeCount']:
                    if type(data[column]) != int:
                        raise InvalidExpression(f"The {column} must be an integer")
                ## Checking erros on surfaces field
                elif column in ['area', 'terraceSurface', 'gardenSurface', 'landSurface']:
                    if type(data[column]) != float:
                        raise InvalidExpression(f"The {column} must be an float")

                ## Checking erros on field
                elif column.startswith('has'):
                    if not re.match(r'^[01]$', str(data[column])):
                        raise InvalidExpression(f"The {column} must be 0 or 1")

                # Checking errors on categorical values:
                elif type(data[column]) == str:
                    if column in self.unique_categorical_values:
                        if data[column] not in self.unique_categorical_values[column]:
                            raise InvalidExpression(f"The value {data[column]} for the field {column} is not valid")
                    else:
                        raise InvalidExpression(f"The value of the field {column} is not valid")
                else:
                    raise InvalidExpression(f"The value given '{data[column]}' must be an string")
            else:
                missing += column + ' '
        if missing.__len__() > 0:
            raise InvalidExpression(f"Fields missing: {missing}")

