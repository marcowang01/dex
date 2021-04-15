from flask import Flask, render_template, url_for, request
import joblib
import numpy as np

import pickle
import pandas as pd
import csv
import time
from tqdm import tqdm
from scipy import sparse
import config
from text_classification import classify as c
tqdm.pandas(desc="progress-bar")

app = Flask(__name__)

# TODO: error checking for symbols etc.
# TODO: grab date and use that in the analysis


@app.route('/', methods=['GET'])
def home():
    return render_template('front.html')


@app.route('/results', methods=['POST', 'GET'])
def predict():

    if request.method == 'POST':
        speech = request.form['article']
        symbol = request.form['symbol']
        _speech = [speech]

        text_counts, data, stock_symbol, cv = c.preprocess(symbol, _speech)
        prediction = c.classify(text_counts, data, stock_symbol, cv)

        tags = ['g', 'b']

        g_prediction = F"{prediction[0][0]:.2%}"
        b_prediction = F"{prediction[0][1]:.2%}"

        best = max(prediction[0][1], prediction[0][0])
        prediction = tags[np.where(prediction[0] == best)[0][0]]

        return render_template('result.html',
                               g_prediction=g_prediction,
                               b_prediction=b_prediction,
                               prediction=prediction)

    return render_template('result.html',
                           g_prediction="0.00%",
                           b_prediction="0.00%",
                           prediction="No predictions yet! Enter an article!")



if __name__ == '__main__':
    app.run(debug=True)
