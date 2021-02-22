#!/usr/bin/env python
# coding: utf-8

import time
import scipy
import string 
import pickle
import random
import argparse
import itertools
import numpy as np
import pandas as pd

from pprint import pprint

from tqdm.notebook import tqdm

from scipy.stats import spearmanr

from gensim.models import KeyedVectors

from sentence_transformers import models, SentenceTransformer

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor, LogisticRegression, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import RegexpTokenizer



parser = argparse.ArgumentParser(description='Computing text scores for MediaEval 2019')

parser.add_argument("-d", "--video_descriptions_path", type=str, help="Path to the CSV file containing video IDs and corresponding description(s)", default='me_2020/me19_training_data.csv')
# parser.add_argument("-d", "--deep_caption_path", type=str, help="Path to the file containing deep captions (in the same order as the training data)", default='me_2020/deep_captions.txt')
# parser.add_argument("-s", "--video_scores_path", type=str, help="Path to the CSV file containing ground-truth scores.", default="me_2020/scores_v2.csv")
parser.add_argument("-wv", "--word_embeddings_path", type=str, help="Path to word embeddings (e.g. GloVe)", default=None)
parser.add_argument('--save_model', action='store_true', default=False)

args = parser.parse_args()



me19 = pd.read_csv(args.video_descriptions_path)
me19.head()


X19_tfidf = []
X19_w2v   = []
Y19 = []

for i, entry in tqdm(me19.iterrows(), total=len(me19)):
    text = tokenize(entry['text'])
    y = (entry['short_term'], entry['long_term'])
    
    x_tfidf = vectorizer.transform([text]).toarray()[0]
    words = [word for word in text.split(' ') if word in w2v]
    x_w2v = np.zeros([300]) if not words else np.mean([w2v[word] for word in words], axis=0)
    
    assert(x_tfidf.shape == (991,))
    assert(x_w2v.shape == (300,))
    
    X19_tfidf.append(x_tfidf)
    X19_w2v.append(x_w2v)
    Y19.append(y)



X19_tfidf = np.array(X19_tfidf)
X19_w2v = np.array(X19_w2v)
Y19 = np.array(Y19)

X19_tfidf.shape, X19_w2v.shape, Y19.shape


corpus19 = [l.lower() for l in  me19['text'].values]

bert19_embeddings = []
for i in tqdm(range(10)):
    embeddings = model.encode(corpus19[i*800: (i+1)*800])
    bert19_embeddings.append(embeddings)

bert19_embeddings = np.concatenate(bert19_embeddings)


X19 = {'tfidf': X19_tfidf, 'w2b':X19_w2v, 'bert': bert19_embeddings}
Y19_st = Y19[:, 0]
Y19_lt = Y19[:, 1]


X = {'tfidf': X_tfidf, 'w2v':X_w2v, 'bert1': bert1_embeddings, 'bert2': bert2_embeddings}
Y_st = Y[:, 0]
Y_lt = Y[:, 1]
me19_st = {}
for k in X19:
    # if k == 'tfidf': continue
    print(k)
    me19_st[k] = {}
    for regressor in enumerate_models(regression_models):
        model_name = str(regressor)
        me19_st[model_name] = []
        print('Training', model_name.split('(')[0], '..')
                
        t = time.time()
        X_train, X_test = X19[k], X[k]
        y_train, y_test = Y19_st, Y_st
        
        if model_name.startswith('SVR'):
            X_train, y_train = X_train[:600], y_train[:600]
        
        # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        me19_st[k][model_name] = (y_pred, y_test)
        print()
        print(f'done! ({(time.time() - t):2} secs)')
        print('Spearman:', round(spearman(y_pred, y_test), 3), '\n')
        t = time.time()
        
me19_lt = {}
for k in X19:
    # if k == 'tfidf': continue
    print(k)
    me19_lt[k] = {}
    for regressor in enumerate_models(regression_models):
        model_name = str(regressor)
        me19_st[model_name] = []
        print('Training', model_name.split('(')[0], '..')
        t = time.time()
        X_train, X_test = X19[k], X[k]
        y_train, y_test = Y19_lt, Y_lt
        
        # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        me19_lt[k][model_name] = (y_pred, y_test)
        print()
        print(f'done! ({(time.time() - t):2} secs)')
        print('Spearman:', round(spearman(y_pred, y_test), 3), '\n')
        t = time.time()