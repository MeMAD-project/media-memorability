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

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor, LogisticRegression, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import RegexpTokenizer



parser = argparse.ArgumentParser(description='Computing text scores for MediaEval 2020')

parser.add_argument("-d", "--video_descriptions_path", type=str, help="Path to the CSV file containing video IDs and corresponding description(s)", default='me_2020/official_video_descriptions.csv')
parser.add_argument("-c", "--deep_caption_path", type=str, help="Path to the file containing deep captions (in the same order as the training data)", default='me_2020/deep_captions.txt')
parser.add_argument("-s", "--video_scores_path", type=str, help="Path to the CSV file containing ground-truth scores.", default="me_2020/scores_v2.csv")
parser.add_argument("-t", "--test_set_path", type=str, help="Path to the CSV file containing video descriptions of the testset.", default="me_2020/test_text_descriptions.csv")
parser.add_argument("-r", "--results_path", type=str, help="Path to where to save the results for short and long term predictions.", default="me_2020/test_text_descriptions.csv")
parser.add_argument("-wv", "--word_embeddings_path", type=str, help="Path to word embeddings (e.g. GloVe)", default=None)
parser.add_argument('--save_model', action='store_true', default=False)

args = parser.parse_args()


# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()
spearman = lambda x,y: spearmanr(x, y).correlation

if args.word_embeddings_path:
    w2v = KeyedVectors.load_word2vec_format(args.word_embeddings_path)

else:
    w2v = pickle.load(open('../conceptnet/glove.6B/glove.w2v.6B.300d.pickle', 'rb'))

stopwords = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would"]

def tokenize(s):
    numbers = {'2': 'two', '3': 'three', '4': 'four'}
    s = ''.join(c for c in s if c not in string.punctuation or c == ' ').lower()
    t = RegexpTokenizer(r'\w+').tokenize(s)
    t = [lemmatizer.lemmatize(w) if w not in numbers else numbers[w] for w in t if w not in stopwords]
    return ' '.join(t)

def tokenize2(s): # keep stopwords and don't lemmatize
    numbers = {'2': 'two', '3': 'three', '4': 'four'}
    s = ''.join(c for c in s if c not in string.punctuation or c == ' ').lower()
    t = RegexpTokenizer(r'\w+').tokenize(s)
    t = [w if w not in numbers else numbers[w] for w in t]
    return ' '.join(t)

def tokenize3(s): # remove duplicates
    numbers = {'2': 'two', '3': 'three', '4': 'four'}
    s = ''.join(c for c in s if c not in string.punctuation or c == ' ').lower()
    t = RegexpTokenizer(r'\w+').tokenize(s)
    t = [w if w not in numbers else numbers[w] for w in t if w not in stopwords]
    return ' '.join(set(t))




df_text = pd.read_csv(args.video_descriptions_path)
df_scores = pd.read_csv(args.video_scores_path)


# For videos having multiple descriptions
text_concat = df_text[['video_id','description']].groupby(['video_id'])['description'].transform(lambda x: ' '.join(x)).drop_duplicates()

deep_captions = [ l[8:].strip() for l in open(args.deep_caption_path)]

df_data = df_scores.copy()
df_data['text'] = text_concat.values
df_data['short_term'] = df_data['part_1_scores']
df_data['long_term'] = df_data['part_2_scores']
df_data['deep_caption'] = deep_captions[:590]
df_data = df_data[['video_id', 'text', 'deep_caption', 'short_term', 'long_term']]


df_data['content'] = df_data['text'] + '  ' + df_data['deep_caption']


corpus = df_data.text.values
corpus_tokenized = [tokenize3(s) for s in corpus]
#corpus_tokenized2 = [tokenize2(s) for s in corpus]
#corpus_tokenized3 = [tokenize3(s) for s in corpus]

vectorizer = TfidfVectorizer(min_df=4, stop_words='english', ngram_range=(1, 2))
vectorizer.fit(corpus_tokenized)

X_train = vectorizer.transform(corpus)

print(X_train.shape)
print(X_train.toarray())


X_tfidf = []
X_w2v   = []
Y = []

for i, entry in tqdm(df_data.iterrows(), total=len(df_data)):
    text = tokenize(entry['text'])
    y = (entry['short_term'], entry['long_term'])
    
    x_tfidf = vectorizer.transform([text]).toarray()[0]
    words = [word for word in text.split(' ') if word in w2v]
    x_w2v = np.zeros([300]) if not words else np.mean([w2v[word] for word in words], axis=0)
    
    X_tfidf.append(x_tfidf)
    X_w2v.append(x_w2v)
    Y.append(y)

X_tfidf = np.array(X_tfidf)
X_w2v = np.array(X_w2v)
Y = np.array(Y)

X_tfidf.shape, X_w2v.shape, Y.shape


# model = SentenceTransformer('distiluse-base-multilingual-cased')
# sbert1 = SentenceTransformer('distiluse-base-multilingual-cased')
# sbert2 = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# bert1_embeddings = sbert1.encode(corpus_tokenized2)
# bert2_embeddings = sbert2.encode(corpus_tokenized2)

def enumerate_models(models):
    instances = []
    for model_name, (model, hyperparameters) in models.items():
        configs = {}
        if len(hyperparameters) > 0:
            params, vals = list(hyperparameters.keys()), list(hyperparameters.values())
            configs = [dict(zip(params, vv)) for vv in list(itertools.product(*vals))]
            for config in configs:
                m = model(**config)
                instances.append(m)
        else:
            instances.append(model())
    return instances


regression_models = {
    # 'LogisticRegression': (LogisticRegression, {"C": [1e3, 1, 1e-3], "penalty": ['l1', 'l2', 'elasticnet']}),
    # 'LinearRegression': (LinearRegression, {}),
    # 'MLPRegressor': (MLPRegressor, {'alpha': [1e-3,  1e-7], 'hidden_layer_sizes': [(10,), (100,)]}), # 1e-5,, (50,), 
    # 'SGDRegressor': (SGDRegressor, {'alpha': [0.0001, 0.1,]}),
    # 'SVR': (SVR, {'kernel': ['linear', 'rbf'], "C": [1e-3, 1e-4, 1e-5, 1e-7], "gamma": ["scale"]})
    'SVR': (SVR, {'kernel': ['linear',], "C": [1e-3, 1e-5, 1e-7], "gamma": ["scale"]})
}
len(enumerate_models(regression_models))


X = {'w2v':X_w2v} # 'tfidf': X_tfidf, 'w2v':X_w2v, 'bert1': bert1_embeddings, 'bert2': bert2_embeddings}
Y_st = Y[:, 0]
Y_lt = Y[:, 1]


folds = {}
print('Short term memorability prediction:'.upper())

for k in X:
    folds[k] = {}
    print('\nFeatures:', k.upper(), '\n')
    for regressor in enumerate_models(regression_models):
        model_name = str(regressor)
        folds[k][model_name] = []
        kf = KFold(n_splits=6, random_state=42)
        print('Training', model_name, '..')
        for i, (train_index, test_index) in enumerate(kf.split(X[k])):
            print('Fold #'+ str(i), end='.. ')
            t = time.time()
            X_train, X_test = X[k][train_index], X[k][test_index]
            y_train, y_test = Y_st[train_index], Y_st[test_index]
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            folds[k][model_name].append((y_pred, y_test))
            print(f'done! ({(time.time() - t):.2} secs). Spearman: {spearman(y_pred, y_test):.2}')
            
            t = time.time()


folds_lt = {}
print('Long term memorability prediction:'.upper())

for k in X:
    folds_lt[k] = {}
    print('\nFeatures:', k.upper(), '\n')
    for regressor in enumerate_models(regression_models):
        model_name = str(regressor)
        folds_lt[k][model_name] = []
        kf = KFold(n_splits=6, random_state=42)
        print('Training', model_name, '..')
        for i, (train_index, test_index) in enumerate(kf.split(X[k])):
            print('Fold #'+ str(i), end='.. ')
            t = time.time()
            X_train, X_test = X[k][train_index], X[k][test_index]
            y_train, y_test = Y_lt[train_index], Y_lt[test_index]
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            folds_lt[k][model_name].append((y_pred, y_test))
            print(f'done! ({(time.time() - t):.2} secs). Spearman: {spearman(y_pred, y_test):.2}')
            
            t = time.time()

            
for term, all_folds in [('Short term', folds), ('Long term', folds_lt), ('Long Short term', folds_lt)]:
    print(term.upper())
    for embedding in all_folds:
        print('  USING', embedding)
        for i, model_name in enumerate(all_folds[embedding]):
            # print('    ', (model_name.split('(')[0] + ' ' + str(i+1)).ljust(18), '\t', end=' ')
            print('    ', model_name, '\t', end=' ')
            # print(', '.join([str(spearman(y_p, y_t)) for y_p, y_t in all_folds[embedding][model_name]]))
            if term == 'Long Short term':
                sps = [spearman(ys_p, yl_t) for (ys_p, ys_t), (yl_p, yl_t)
                       in zip(folds[embedding][model_name], folds_lt[embedding][model_name])]
            else:
                sps = [spearman(y_p, y_t) for y_p, y_t in all_folds[embedding][model_name]]
            print(round(sum(sps)/len(sps), 4))


ismail_st = folds['w2v']["SVR(C=1e-05, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='scale',\n    kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)"]
ismail_lt = lt_folds['w2v']["SVR(C=1e-05, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='scale',\n    kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)"]



# # Combining everything

# In[51]:




## Testset results
df_text_test = pd.read_csv(args.test_set_path)


test_text_concat = df_text_test[['video_id','description']].groupby(['video_id'])['description'].transform(lambda x: '  '.join(x)).drop_duplicates()


df_test = pd.DataFrame({"video_id": df_text_test.video_id.unique(),
                        "text": test_text_concat,})

X_test_tfidf = []
X_test_w2v   = []

for i, entry in tqdm(df_test.iterrows(), total=len(df_test)):
    text = tokenize(entry['text'])
    
    x_tfidf = vectorizer.transform([text]).toarray()[0]
    words = [word for word in text.split(' ') if word in w2v]
    x_w2v = np.zeros([300]) if not words else np.mean([w2v[word] for word in words], axis=0)
    
    X_test_tfidf.append(x_tfidf)
    X_test_w2v.append(x_w2v)


X_train = X['w2v']
y_train = Y_st
svr_model = SVR(C=1e-05, gamma='scale', kernel='linear')
svr_model.fit(X_train, y_train)
ismail_st_test = svr_model.predict(X_test_w2v)


X_train = X['w2v']
y_train = Y_lt
svr_model_lt = SVR(C=1e-05, gamma='scale', kernel='linear')
svr_model_lt.fit(X_train, y_train)
ismail_lt_test = svr_model_lt.predict(X_test_w2v)


ismail_preds = pd.DataFrame({'video_id': df_text_test.video_id.unique(),
                             'short_term': ismail_st_test,
                             'long_term': ismail_lt_test})

ismail_preds.to_csv('me_2020/ismail_testset_preds_svr_1e-05_scale_lin.csv')
# ismail_preds = pd.read_csv('me_2020/ismail_testset_preds_svr_1e-05_scale_lin.csv')
