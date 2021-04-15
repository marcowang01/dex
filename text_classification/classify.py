from datetime import datetime, timedelta
import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from nltk import sent_tokenize
from tqdm import tqdm
from scipy import sparse
import pickle
import config
from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import random
from text_classification import StockAPI
import time

classifiers = {
    "BernoulliNB": BernoulliNB(),
    "ComplementNB": ComplementNB(),
    "MultinomialNB": MultinomialNB(alpha=0.3),
    # "KNeighborsClassifier": KNeighborsClassifier(),
    # "DecisionTreeClassifier": DecisionTreeClassifier(),
    # "RandomForestClassifier": RandomForestClassifier(),
    # "LogisticRegression": LogisticRegression(max_iter=10000),
    # "SGDClassifier": SGDClassifier(),
    # "AdaBoostClassifier": AdaBoostClassifier(),
    # "MLPClassifier": MLPClassifier(max_iter=1000),
}


TAGS = ['g', 'b']


def preprocess(stock_symbol, new_speech=None):
    if config.DO_GET_QUOTES and config.DO_TRAINING:
        # tags the speeches by stock quotes
        # creates quote object from StockAPI.py
        stock = StockAPI.Quote(stock_symbol, '4. close')
        data = read_data(stock_symbol, stock)
    else:
        data = read_data(stock_symbol)

    print("\nTable info")
    data.info()  # prints table structure to terminal

    if config.DO_TRAINING:
        tokens = RegexpTokenizer(r'[a-zA-Z]+')
        cv = CountVectorizer(tokenizer=tokens.tokenize, stop_words="english", ngram_range=(1, 2))

        print("\nGenerating bag of words:")
        text_counts = cv.fit_transform(data['content'])  # creates a doc-term matrix
        print(F"Matrix size: {text_counts.shape}")
        pickle.dump(cv, open("./pickles/cv.sav", 'wb'))

        if config.DO_INTEGRATION:
            text_counts = integrate_db("../dataset/master_dict/master_dict_filtered.csv", data, text_counts, cv)
    else:
        cv = pickle.load(open("text_classification/pickles/cv.sav", 'rb'))

        print("\nGenerating bag of words:")
        text_counts = cv.transform(new_speech)

        if config.DO_INTEGRATION:
            text_counts = integrate_db("dataset/master_dict/master_dict_filtered.csv", data, text_counts, cv)



    return text_counts, data, stock_symbol, cv


def classify(text_counts, data, stock_symbol, cv):
    if config.DO_TRAINING:
        RANDOM_STATE: int = config.RANDOM_STATE
        X_train, X_test, y_train, y_test = train_test_split(
            text_counts, data['tag'], test_size=0.3, random_state=RANDOM_STATE)

        print("\nTraining Classifier:")
        # 0: accuracy, 1: name of clf: 2: clf itself
        highest_score = [0, "", None]
        # trains all classifiers within classifiers dictionary
        for name, sklearn_clf in classifiers.items():
            start = time.time()
            clf = sklearn_clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            end = time.time()

            print(f"{name} ({(end - start) / 60:.3} min)")
            accuracy = metrics.accuracy_score(y_test, y_pred)
            # keep track of highest accuracy to be saved
            if accuracy > highest_score[0]:
                highest_score[0] = accuracy
                highest_score[1] = name
                highest_score[2] = clf

            print(F"{accuracy:.2%} - {stock_symbol}")
            # print(classification_report(y_test, y_pred, target_names=TAGS))

        log_result(highest_score[1], highest_score[0], stock_symbol)
        pickle.dump(highest_score[2],
                    open(F"./pickles/{highest_score[1]}_{highest_score[0]:.2%}_{stock_symbol}.sav",
                         'wb'))
        return None
    else:
        CLF_NAME = config.CLF_NAME
        PERCENT = config.PERCENT

        clf = pickle.load(open(f"text_classification/pickles/{CLF_NAME}_{PERCENT}_{stock_symbol}.sav", 'rb'))

        return clf.predict_proba(text_counts)


def integrate_db(db_path, data, text_counts, cv: CountVectorizer):
    length = text_counts.shape[0]

    if config.DO_TRAINING:
        feature_list = cv.get_feature_names()
        # translates list of features in dict {word => index}
        feature_dict = {feature_list[i]: i for i in range(0, len(feature_list))}
        pickle.dump(feature_dict, open("./pickles/feature_dict.sav", 'wb'))
    else:
        feature_dict = pickle.load(open("text_classification/pickles/feature_dict.sav", 'rb'))

    lil_tc = sparse.lil_matrix(text_counts)  # converts text counts from csr matric to lil matrix to increase efficiency

    with open(db_path, 'r') as f:
        reader = csv.DictReader(f)
        pbar = tqdm(total=length)  # makes a new progress bar

        # iterate through all documents
        for doc_i in range(length):
            # iterate through each word in filtered_master_dict
            for row in reader:
                # for positive words check if they exist in text_counts' features
                if data['tag'][doc_i] == 'g':
                    if row['Positive'] != 'empty' and row['Positive'] in feature_dict:
                        word_i = feature_dict[row['Positive']]
                        # multiplies entry by frequency in master_dict filtered if document is tagged as 'g'
                        lil_tc[doc_i, word_i] *= float(row['Pos Freq']) * 10
                elif data['tag'][doc_i] == 'b':
                    if row['Negative'] != 'empty' and row['Negative'] in feature_dict:
                        word_i = feature_dict[row['Negative']]
                        lil_tc[doc_i, word_i] *= float(row['Neg Freq']) * 10

            pbar.update(1)
        pbar.close()

    return sparse.csr_matrix(lil_tc) # converts lil matrix back to csr


def read_data(stock_symbol, stock: StockAPI.Quote = None):
    if config.DO_GET_QUOTES and config.DO_TRAINING:
        print("reading speeches...")
        data = pd.read_json("../dataset/Fed/powell_data.json")

        tags = []
        print("reading quotes...")
        for i in tqdm(data.index):
            date_str = str(data["date"][i])
            Y = date_str[0:4]
            m = date_str[4:6]
            d = date_str[6:8]

            # converts date into correct format
            speech_date = datetime.strptime(F"{Y}-{m}-{d}", '%Y-%m-%d')
            date1 = speech_date - timedelta(days=1)
            date1 = datetime.strftime(date1, '%Y-%m-%d')
            date2 = datetime.strftime(speech_date, '%Y-%m-%d')

            # grab stock quotes from dates
            delta = stock.lookup(date1, date2)

            # if negative --> 'b' ...
            if delta > 0:
                tag = TAGS[0]
            else:
                tag = TAGS[1]

            tags.append(tag)
        # insert new column into dataframe object
        data.insert(5, "tag", tags, True)

        # tokenize dataframe content into sentences
        sentence_list = []
        for i in range(len(data)):
            sentences = sent_tokenize(data['content'][i])
            for s in sentences:
                temp_dict = {
                    'date': data['date'][i],
                    'title': data['title'][i],
                    'content': s,
                    'tag': data['tag'][i]
                }
                sentence_list.append(temp_dict)
        # converts list of dictionary into dataframe object
        data = pd.DataFrame(sentence_list)

        data.to_pickle(f"../dataset/Fed/powell_{stock_symbol}.pkl")
    else:
        data = pickle.load(open(f"dataset/Fed/powell_{stock_symbol}.pkl", 'rb'))

    print("\n5 docs:")
    for i in [0, 1000, 2000, 3000, 4000, 5000]:
        print(F"{data['date'][i]}: {data['tag'][i]}  \t|  {data['title'][i]}")

    return data


# writes results into file
def log_result(clf_name, score, stock_symbol):
    with open("./results.txt", "a") as f:
        f.write(F"{stock_symbol}: {score:.2%} - {clf_name}\n")


if __name__ == '__main__':

    tickers = ["PLUG"]
    for t in tickers:
        text_counts, data, stock_symbol, cv = preprocess(t)
        classify(text_counts, data, stock_symbol, cv)
