import pandas as pd
import numpy as np

from exceptions.InvalidExpression import InvalidExpression


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    This method will clean the dataframe deleting duplicated rows,
    fixing errors, eliminating outliers. If visualize_flag set
    will plot correlation map and relationships between
     the variables and the target price
    :param df: dataframe to clean
    :param visualize_flag: Will plot all relationships with variable price
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

    # Deleting variable 'typeProperty', keeping 'subtypeProperty'

    if 'price' in df.columns:
        # Dropping rows with price as NaN values
        df = df[df["price"].notna()]
        df = df[df["area"].notna()]

        # Dropping duplicated values
        df = df.drop_duplicates(
            subset=["area", "price"], keep="last"
        )

        # Dropping outliers
        df = df[df["price"] < 6000000]
        df = df[df["area"] < 1350]

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

    # Creating new variable adding outside surface
    df.terraceSurface.fillna(0, inplace=True)
    df.gardenSurface.fillna(0, inplace=True)
    df["outsideSpace"] = (
            df["terraceSurface"] + df["gardenSurface"]
    )

    # Filling nan values to 0 for facadeCount
    df.facadeCount.fillna(0, inplace=True)

    # Deleting least correlated, reformatted and constant columns
    to_drop = ['kitchenType', 'typeProperty', 'typeSale', 'subtypeSale', 'isFurnished'
               'terraceSurface', 'gardenSurface']

    for feature in to_drop:
        if feature in df.columns.to_list():
            del df[feature]

    # Transform  variables into features
    features = [
        "postalCode",
        "buildingCondition",
        "subtypeProperty",
        "hasFireplace",
        "hasSwimmingPool",
        "hasGarden",
        "hasTerrace",
        "hasFullyEquippedKitchen",
    ]

    print(df.columns.to_list())

    for feature in features:
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

    df = rescale(df)

    return df.reset_index(drop=True)


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


def checkErrors(data: dict):
    columns = ['postalCode', 'subtypeProperty', 'price', 'bedroomsCount', 'area', 'hasFullyEquippedKitchen', 'hasFireplace', 'hasTerrace', 'terraceSurface', 'hasGarden', 'gardenSurface', 'landSurface', 'facadeCount', 'hasSwimmingPool', 'buildingCondition']

    missing = ''

    # First check if there is any value missing
    for column in columns:
        if column in data:
            if column == 'postalCode':
                if 0 > data[column] or data[column]> 9999:
                    raise InvalidExpression(f"The zip code should be a number between 0 and 9999.", status_code=200)

        else:
            missing += column + ' '
    if missing.__len__() > 1:
        return {"error": f"3 fields expected (salary, bonus, taxes). You forgot: {missing}."}
    else:
        return {"result", "all values here"}
