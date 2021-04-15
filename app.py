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

        text_counts, data, stock_symbol, cv = c.preprocess(symbol, speech)
        g, b, best = c.classify(text_counts, data, stock_symbol, cv)

        g_prediction = F"{g:.2%}"
        b_prediction = F"{b:.2%}"

        return render_template('result.html',
                               g_prediction=g_prediction,
                               b_prediction=b_prediction,
                               prediction=best)

    return render_template('result.html',
                           g_prediction="0.00%",
                           b_prediction="0.00%",
                           prediction="No predictions yet! Enter an article!")



if __name__ == '__main__':
    app.run(debug=True)
