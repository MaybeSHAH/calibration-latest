#!/usr/bin/python3

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib


app = Flask(__name__)
CORS(app)

# Load the model
model = joblib.load('./models/LCS_Calibration_Model_Mumbai.pkl')
apportion_model = joblib.load('./models/regressor.pkl')
# model = pickle.load(open('./models/LCS_Calibration_Model_Mumbai.pkl','rb'))
@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # print(data)
    # Make prediction using model loaded from disk as per the data.
    pred = model.predict([np.array(data)])
    # print("X-test", X_test)
    # print(pred)
    # prediction = model.predict([[np.array(data['exp'])]])
    # Take the first value of prediction
    # output = float("%.2f" % pred[0])
    return jsonify(float("%.2f" % pred[0]))
@app.route('/apportion',methods=['POST'])
def apportion():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    pred = apportion_model.predict([np.array(data)])
    # print("X-test", X_test)
    print(pred)
    # prediction = model.predict([[np.array(data['exp'])]])
    # Take the first value of prediction
    # output = float("%.2f" % pred[0][0])
    formatted_list = ["%.2f" % elem for elem in pred[0]]
    # make object here and then send { a:1, b:2, c:3 }
    return jsonify(formatted_list)
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000, debug=True)
