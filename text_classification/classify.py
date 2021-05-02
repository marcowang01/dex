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
from StopWords_Generic import stopwords

classifiers = [("BernoulliNB", BernoulliNB()),
        ("ComplementNB", ComplementNB()),
        ("MultinomialNB", MultinomialNB()),
        # ("KNeighborsClassifier", KNeighborsClassifier()),
        # ("DecisionTreeClassifier", DecisionTreeClassifier()),
        # ("RandomForestClassifier", RandomForestClassifier()),
        # ("LogisticRegression", LogisticRegression(max_iter=1000)),
        # ("SGDClassifier", SGDClassifier()),
        # ("AdaBoostClassifier", AdaBoostClassifier()),
        # ("MLPClassifier", MLPClassifier(max_iter=1000)),
        # ("SVC", SVC()),
        # ("NuSVC", NuSVC()),
        # ("LinearSVC", LinearSVC())
]



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
        cv = CountVectorizer(tokenizer=tokens.tokenize, stop_words='english', ngram_range=(1, 2))

        print("\nGenerating bag of words:")
        text_counts = cv.fit_transform(data['content'])  # creates a doc-term matrix
        print(F"Matrix size: {text_counts.shape}")
        # path = "text_classification/pickles/cv.sav"
        path = "./pickles/cv.sav"
        pickle.dump(cv, open(path, 'wb'))

    else:
        # path = "text_classification/pickles/cv.sav"
        path = "./pickles/cv.sav"
        cv = pickle.load(open(path, 'rb'))

        print("\nGenerating bag of words:")
        # turn new articles into sentences
        sentences = sent_tokenize(new_speech)
        text_counts = cv.transform(sentences)


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
        for tup in classifiers:
            name = tup[0]
            sklearn_clf = tup[1]
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
            print(classification_report(y_test, y_pred, target_names=config.TAGS))

        log_result(highest_score[1], highest_score[0], stock_symbol)
        # path = F"text_classification/pickles/{highest_score[1]}_{highest_score[0]:.2%}_{stock_symbol}.sav"
        path = F"./pickles/{highest_score[1]}_{highest_score[0]:.2%}_{stock_symbol}.sav"
        pickle.dump(highest_score[2], open(path,'wb'))
        return None

    else:
        CLF_NAME = config.CLF_NAME
        PERCENT = config.PERCENT
        print(f"using {CLF_NAME} on {PERCENT}")
        # path = f"text_classification/pickles/{CLF_NAME}_{PERCENT}_{stock_symbol}.sav"
        path = f"./pickles/{CLF_NAME}_{PERCENT}_{stock_symbol}.sav"
        clf = pickle.load(open(path, 'rb'))

        scores = clf.predict_proba(text_counts)
        good_percent = 0
        bad_percent = 0
        length = text_counts.shape[0]
        for sent_pred in scores:
            good_percent += sent_pred[0]
            bad_percent += sent_pred[1]

        good_percent = good_percent / float(length)
        bad_percent = bad_percent / float(length)
        best = "g" if good_percent > bad_percent else "b"

        print(f"Good: {good_percent:.2%}\nBad: {bad_percent:.2%}")

        return good_percent, bad_percent, best


def read_data(stock_symbol, stock: StockAPI.Quote = None):
    if config.DO_GET_QUOTES and config.DO_TRAINING:
        print("reading speeches...")
        # path = "dataset/Fed/powell_data.json"
        path = "./../dataset/Fed/powell_data.json"
        data = pd.read_json(path)

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
                tag = config.TAGS[0]
            else:
                tag = config.TAGS[1]

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
        # path = f"dataset/Fed/powell_{stock_symbol}_df.pkl"
        path = f"./../dataset/Fed/powell_{stock_symbol}_df.pkl"
        data.to_pickle(path)
    else:
        # path = f"dataset/Fed/powell_{stock_symbol}_df.pkl"
        path = f"./../dataset/Fed/powell_{stock_symbol}_df.pkl"
        data = pickle.load(open(path, 'rb'))

    print("\n5 docs:")
    for i in [0, 1000, 2000, 3000, 4000, 5000]:
        print(F"{data['date'][i]}: {data['tag'][i]}  \t|  {data['title'][i]}")

    return data


# writes results into file
def log_result(clf_name, score, stock_symbol):
    with open("./results.txt", "a") as f:
        f.write(F"{stock_symbol}: {score:.2%} - {clf_name}\n")


if __name__ == '__main__':

    for t in config.tickers:
        text_counts, data, stock_symbol, cv = preprocess(t, config.newSpeech)
        classify(text_counts, data, stock_symbol, cv)
