[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<div align = "center">

<h1>Real State Prediction API</h1>
</div>
# ImmoEliza Predictions API
API Machine learning model to predict prices on Belgium's real estate sales.

## Table of contents
[Description](#Description)  
[Installation](#Installation)  
[Usage](#Usage)  
[Output](#Output)  
[How it works](#How-it-works)  
[Examples](#Examples)  
[Authors](#Authors)

## Description
The API return the prediction the price of a propertie in Belgium, based on data scrapped from Immoweb. 
For the predictions our Machine Learning model looks at the relationship between the postal code, the state of the construction, the property subtype (apartment, studio, villa, chalet, ...), and existance of a fireplace, terrace, garden and/or fully equiped kitchen, an estimate of the asking price is made.

The accuracy of the model is pf  85%, which means that there is always a possibility for outliers (less then 15 %).
  
This API has been deployed with heroku under the url: https://api-ie-predictions.herokuapp.com/

## Usage
  
For the predictions, send a `POST` request to https://api-ie-predictions.herokuapp.com/predict with the following parameters:
  
  ```json
{
  "data": {
      "area": float,
      "subpropertyType": Optional['HOUSE', 'VILLA', 'EXCEPTIONAL_PROPERTY', 'APARTMENT_BLOCK',
          'MANSION', 'MIXED_USE_BUILDING', 'BUNGALOW', 'TOWN_HOUSE',
          'FARMHOUSE', 'COUNTRY_COTTAGE', 'MANOR_HOUSE', 'APARTMENT',
          'PENTHOUSE', 'DUPLEX', 'TRIPLEX', 'LOFT', 'FLAT_STUDIO',
          'SERVICE_FLAT', 'GROUND_FLOOR'],
      "bedroomsCount": int,
      "postalCode": int,
      "ladnSurface": float,
      "hasGarden": binary bool,
      "gardenSurface": binary bool,
      "hasFullyEquippedKitchen": binary bool,
      "hasSwimmingPool": binary bool,
      "hasFireplace": binary bool,
      "hasTerrace": bianry bool,
      "terraceSurface": float,
      "facadeCount": int,
      "buildingCondition": Optional["TO_BE_DONE_UP" , "AS_NEW" , "GOOD" , "JUST_RENOVATED" , "TO_RESTORE"]
      }
}
```
When you run the program, you will get: 

- a print of the train and test scores (without and with boost),
- a print of the regressor score,
- a list of data that are to be predicted (based on the test dataframe),
- a list of the predictions themselves,
- some useful plots (the normalized data, the predictions vs the y_test database to show the accuracy, 
the boosted predictions, and the normalized data of the test dataframe, ...)

## How it works
1. DataCleaner
First, the data are cleaned. That means that we drop all the entirely empty rows, string values
are cleaned up, outliers and properties without price and area indication are dropped, duplicates
and columns with the lowest correlation rate are deleted, and some other minor riddances.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To put everything ready for the rest of the process, the variables that remain are transformed into
features.

2. DataRegressor
In the second step, the prediction is prepared. Firstly, the price, area, outside space and land
surface are rescaled. This is done in order to limit the differences and make the model more
effective.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Secondly, the database is split and into a train and test dataframe. The former is used to train the model. A gradient 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;boost is implemented.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In the final step, predictions are made using the test dataset.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;As an important addendum, we created a function that will prepare any new dataset to be pushed through the 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;program and make predictions about the price.


## Author
Jesús Bueno - Project Manager/dev & doc  
