import os

from flask import Flask, request, jsonify
from flask_cors import cross_origin

from exceptions.InvalidExpression import InvalidExpression
from predict.prediction import Predictor

app = Flask(__name__)


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictPrice():
    data = request.get_json()

    # First check if there is any value missing
    response = jsonify(predictor.predict(data['data']))
    # Enable Access-Control-Allow-Origin
    return response


@app.errorhandler(InvalidExpression)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/", methods=['GET'])
def isAlive():
    response = jsonify(message="Alive")
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.errorhandler(404)
def invalid_route(e):
    return "Invalid route."


if __name__ == "__main__":
    predictor = Predictor()
    # You want to put the value of the env variable PORT if it exist (some services only open specifiques ports)
    port = int(os.environ.get('PORT', 5000))
    # Threaded option to enable multiple instances for
    # multiple user access support
    # You will also define the host to "0.0.0.0" because localhost will only be reachable from inside de server.
    app.run(host="0.0.0.0", threaded=True, port=port)

