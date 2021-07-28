from flask import Flask, request, jsonify

from exceptions.InvalidExpression import InvalidExpression
from predict.prediction import Predictor
from preprocessing.cleaning_data import checkErrors

app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def predictPrice():
    data = request.get_json()

    checkErrors(data)
    # First check if there is any value missing
    return jsonify(predictor.predict(data))


@app.errorhandler(InvalidExpression)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/", methods=['GET'])
def isAlive():
    return 'Hello'


@app.errorhandler(404)
def invalid_route(e):
    return "Invalid route."


if __name__ == "__main__":
    predictor = Predictor()
    app.run()
