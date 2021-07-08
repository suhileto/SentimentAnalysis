import numpy as np
import pandas as pd
import string
import nltk

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def preprocessing(text, stopwords_path):
    stopwords = pd.read_csv(stopwords_path)
    stpword = stopwords.values
    for i in range(len(text)):
        no_punctuation = [char for char in text[i] if char not in string.punctuation]
        no_punctuation = ''.join(no_punctuation)
        text[i] = ' '.join([word for word in no_punctuation.split() if word.lower() not in stpword])
    return text

### Parameters
data_path = 'archive/Train.csv'
stopwords_path = 'Stopwords.txt'
preprocess = False

# read data
data = pd.read_csv(data_path)

"""
# encode labels
labeling = {
    'Olumlu':1, 
    'Olumsuz':0
}

data['label'] = data['label'].apply(lambda x : labeling[x])
"""

# preprocess
if (preprocess):
    X = preprocessing(data['text'].values, stopwords_path)
else:
    X = data["text"].values
y = data.drop("text", axis=1).values
y = y.ravel()

### Training Parameters
model_name_list = ['RandomForest', 'NaiveBayes', 'KNN', 'SVC']
test_size_list = [0.7, 0.6, 0.5, 0.3]
num_of_models, num_of_test_sizes = 4, 4

for model_name_index in range(num_of_models):
    model_name = model_name_list[model_name_index]

    for test_size_index in range(num_of_test_sizes):
        if (model_name == 'RandomForest'):
            model = RandomForestClassifier(max_depth=2, random_state=0)
        elif (model_name == 'NaiveBayes'):
            model = GaussianNB()
        elif (model_name == 'KNN'):
            model = KNeighborsClassifier(n_neighbors=3)
        elif (model_name == 'SVC'):
            model = SVC()

        # split data
        test_size = test_size_list[test_size_index]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # vectorize data
        vect = CountVectorizer()
        X_train = vect.fit_transform(X_train)
        X_test = vect.transform(X_test)

        # transform data to tfidf form
        tfidf = TfidfTransformer()
        X_train = tfidf.fit_transform(X_train)
        X_test = tfidf.transform(X_test)
        X_train = X_train.toarray()
        X_test = X_test.toarray()

        # fit
        model.fit(X_train, y_train)

        # evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print('\n*** ', model_name, " test_size ", str(test_size), " acc ", acc, " f1 ", f1, ' ***')
