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

spearman = lambda x,y: spearmanr(x, y).correlation

#from transformers import AutoTokenizer, AutoModelForSequenceClassification
  
#tokenizer = AutoTokenizer.from_pretrained("fabriceyhc/bert-base-uncased-yahoo_answers_topics")

#model = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-yahoo_answers_topics")



parser = argparse.ArgumentParser(description='Computing text scores for MediaEval 2021')


parser.add_argument("-d", "--video_descriptions_path", type=str, help="Path to the CSV file containing video IDs and corresponding description(s)", default='features_2021/i3d-25-128-avg.pickle')

parser.add_argument("-s", "--video_scores_path", type=str, help="Path to the CSV file containing ground-truth scores.", default="/home/semantic/media-memorability/textual_scores/data_2021/TRECVid Data/training_set/train_scores.csv")
#parser.add_argument("-s", "--video_scores_path", type=str, help="Path to the CSV file containing ground-truth scores.", default="/home/semantic/media-memorability/textual_scores/data_2021/Memento10k Data/training_set/train_scores.csv")


#parser.add_argument("-devscores", "--dev_scores_path", type=str, help="Path to the CSV file containing ground-truth scores dev set.", default="/home/semantic/media-memorability/textual_scores/data_2021/TRECVid Data/training_set/dev_scores.csv")
                    
parser.add_argument("-devscores", "--dev_scores_path", type=str, help="Path to the CSV file containing ground-truth scores dev set.", default="/home/semantic/media-memorability/textual_scores/data_2021/TRECVid Data/Memento10k Data/dev_set/dev_scores.csv")




#parser.add_argument("-t", "--test_set_path", type=str, help="Path to the CSV file containing video s of the testset.", default="/home/semantic/media-memorability/textual_scores/data_2021/TRECVid Data/test_set/test_text_descriptions.csv")

parser.add_argument("-t", "--test_set_path", type=str, help="Path to the CSV file containing video s of the testset.", default="/home/semantic/media-memorability/textual_scores/data_2021/Memento10k Data/test_set/test_text_descriptions.csv")



                    
parser.add_argument("-r", "--results_path", type=str, help="Path to where to save the results for short and long term predictions.", default="data_2021/test_text_descriptions.csv")
parser.add_argument("-wv", "--word_embeddings_path", type=str, help="Path to word embeddings (e.g. GloVe)", default=None)
parser.add_argument('--save_model', action='store_true', default=False)

args = parser.parse_args()



features= pd.read_pickle(args.video_descriptions_path)
#keys = label': '100493', 'source': 'memento', 'index': 493}, {'feature': 'i3d-25-128-avg', 'vector
print(len(features))




df_data = pd.read_csv(args.video_scores_path)
                    
df_data['short_term'] = df_data['scores_raw_short_term']
df_data['short_term_norm'] = df_data['scores_normalized_short_term']

if args.video_scores_path!='/home/semantic/media-memorability/textual_scores/data_2021/Memento10k Data/training_set/train_scores.csv':
    df_data['long_term'] = df_data['scores_raw_long_term']
    df_data = df_data[['video_id', 'short_term', 'long_term','short_term_norm']]
                                                  
else:
    df_data = df_data[['video_id', 'short_term','short_term_norm']]
#df_data['deep_caption'] = deep_captions[:590]



X = []

Y = []

for i, entry in tqdm(df_data.iterrows(), total=len(df_data)):
    if args.video_scores_path!='/home/semantic/media-memorability/textual_scores/data_2021/Memento10k Data/training_set/train_scores.csv':
        y = (entry['short_term'], entry['short_term_norm'],entry['long_term'])
        x = [d['vector'] for d in features if d['index']== entry['video_id'].astype(int) and d['source']=='trecvid']
    else:
        y = (entry['short_term'],entry['short_term_norm'])
        x = [d['vector'] for d in features if d['index']== entry['video_id'].astype(int) and  d['source']=='memento']
    if len(x)!=1:
        print(x)
    X.append(x[0])
    Y.append(y)

X = np.array(X)

Y = np.array(Y)




#bert3_embeddings=np.c_[bert3_embeddings , perplexity]


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
     #'LogisticRegression': (LogisticRegression, {"C": [1e3, 1, 1e-3], "penalty": ['l2', 'elasticnet']}),
     #'LinearRegression': (LinearRegression, {}),
     #'MLPRegressor': (MLPRegressor, {'alpha': [1e-3,  1e-7], 'hidden_layer_sizes': [(10,), (100,)]}), # 1e-5,, (50,), 
     #'SGDRegressor': (SGDRegressor, {'alpha': [0.0001, 0.1,]}),
     #'SVR': (SVR, {'kernel': ['linear', 'rbf'], "C": [1e-3, 1e-4, 1e-5, 1e-7], "gamma": ["scale"]})}
     #'SVR': (SVR, {'kernel': ['linear'], "C": [1e-3, 1e-4, 1e-5], "gamma": ["scale"]})}
    'SVR': (SVR, {'kernel': ['linear'], "C": [1e-3], "gamma": ["scale"]})}
len(enumerate_models(regression_models))


X = {'X': X}#,'tfidf': X_tfidf, 'w2v':X_w2v, 'bert1': bert1_embeddings, 'bert2': bert2_embeddings,'bert3': bert3_embeddings}
if args.video_scores_path!='/home/semantic/media-memorability/textual_scores/data_2021/Memento10k Data/training_set/train_scores.csv':                                                       
    Y_st= Y[:, 0]
    Y_st_norm = Y[:, 1]                                                       
    Y_lt= Y[:, 2]
                                                           

                                                           
else:
    Y_st=Y[:, 0]
    Y_st_norm = Y[:, 1] 
    
df_data['short_term_pred']=0

folds = {}
print('Short term raw memorability prediction:'.upper())

for k in X:
    folds[k] = {}
    print('\nFeatures:', k.upper(), '\n')
    for regressor in enumerate_models(regression_models):
        model_name = str(regressor)
        folds[k][model_name] = []
        kf = KFold(n_splits=6, random_state=42,shuffle=True)
        print('Training', model_name, '..')
        for i, (train_index, test_index) in enumerate(kf.split(X[k])):
            print('Fold #'+ str(i), end='.. ')
            t = time.time()
            X_train, X_test = X[k][train_index], X[k][test_index]
            y_train, y_test = Y_st[train_index], Y_st[test_index]
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            for j in range(len(test_index)):
                df_data.loc[df_data.index== test_index[j], 'short_term_pred']=y_pred[j]
                #df.loc[df['set_of_numbers'] == 5, 'set_of_numbers'] = 555
            folds[k][model_name].append((y_pred, y_test))
            print(f'done! ({(time.time() - t):.2} secs). Spearman: {spearman(y_pred, y_test):.2}')
            
            t = time.time()

df_data['short_term_norm_pred']=0         

folds_st_norm = {}
print('Short term norm memorability prediction:'.upper())

for k in X:
    folds_st_norm [k] = {}
    print('\nFeatures:', k.upper(), '\n')
    for regressor in enumerate_models(regression_models):
        model_name = str(regressor)
        folds_st_norm [k][model_name] = []
        kf = KFold(n_splits=6, random_state=42,shuffle=True)
        print('Training', model_name, '..')
        for i, (train_index, test_index) in enumerate(kf.split(X[k])):
            print('Fold #'+ str(i), end='.. ')
            t = time.time()
            X_train, X_test = X[k][train_index], X[k][test_index]
            y_train, y_test = Y_st_norm[train_index], Y_st_norm[test_index]
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            for j in range(len(test_index)):
                df_data.loc[df_data.index== test_index[j], 'short_term_norm_pred']=y_pred[j]
            folds_st_norm [k][model_name].append((y_pred, y_test))
            print(f'done! ({(time.time() - t):.2} secs). Spearman: {spearman(y_pred, y_test):.2}')
            
            t = time.time()

                                                           
                                                           
                                                           
                                                           
                                                           
df_data['long_term_pred']=0                                                     

folds_lt = {}
print('Long term memorability prediction:'.upper())
if args.video_descriptions_path!='/home/semantic/media-memorability/textual_scores/data_2021/Memento10k Data/training_set/train_scores.csv':
    for k in X:
        folds_lt[k] = {}
        print('\nFeatures:', k.upper(), '\n')
        for regressor in enumerate_models(regression_models):
            model_name = str(regressor)
            folds_lt[k][model_name] = []
            kf = KFold(n_splits=6, random_state=42,shuffle=True)
            print('Training', model_name, '..')
            for i, (train_index, test_index) in enumerate(kf.split(X[k])):
                print('Fold #'+ str(i), end='.. ')
                t = time.time()
                X_train, X_test = X[k][train_index], X[k][test_index]
                y_train, y_test = Y_lt[train_index], Y_lt[test_index]
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                for j in range(len(test_index)):
                    df_data.loc[df_data.index== test_index[j], 'long_term_pred']=y_pred[j]
                folds_lt[k][model_name].append((y_pred, y_test))
                print(f'done! ({(time.time() - t):.2} secs). Spearman: {spearman(y_pred, y_test):.2}')

                t = time.time()
                                                           
                                                          

                                                      
print(df_data)                                                         
                                                           
df_data.to_csv('visual_pred_training_set_trecvid.csv')                                                          
                                                           
                                                           
                                                           
if args.video_descriptions_path!='/home/semantic/media-memorability/textual_scores/data_2021/Memento10k Data/training_set/train_scores.csv':                                                         
    types_of_scores=  [('Short term raw', folds), ('Short term norm', folds_st_norm),('Long term', folds_lt), ('Long Short term', folds_lt) ]
else:
    types_of_scores=  [('Short term raw', folds), ('Short term norm', folds_st_norm)]                                           
                                                           
            
for term, all_folds in types_of_scores:
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
                                                           
                #ALSO TRY WITH 
                #sps = [spearman(ys_p, yl_t) for (ys_p, ys_t), (yl_p, yl_t)
                       #in zip(folds_st_norm[embedding][model_name], folds_lt[embedding][model_name])]
            else:
                sps = [spearman(y_p, y_t) for y_p, y_t in all_folds[embedding][model_name]]
            print(round(sum(sps)/len(sps), 4))


#ismail_st = folds['w2v']["SVR(C=1e-05, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='scale',\n    kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)"]
#ismail_lt = lt_folds['w2v']["SVR(C=1e-05, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='scale',\n    kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)"]



# # Combining everything

# In[51]:


"""

## Testset and dev results
df_test = pd.read_csv(args.test_set_path)
df_test=df_test.drop_duplicates(subset=['video_id'])


#df_test = pd.DataFrame({"video_id": df_text_test.video_id.unique(),
                        "text": test_text_concat,})





df_dev = pd.read_csv(args.dev_scores_path)
                    
df_dev['short_term'] = df_dev['scores_raw_short_term']
df_dev['short_term_norm'] = df_dev['scores_normalized_short_term']

if args.video_scores_path!='/home/semantic/media-memorability/textual_scores/data_2021/Memento10k Data/training_set/train_scores.csv':
    df_dev['long_term'] = df_dev['scores_raw_long_term']
    df_dev = df_dev[['video_id', 'short_term', ' long_term','short_term_norm']]
                                                  
else:
    df_dev = df_dev[['video_id', 'short_term','short_term_norm']]




X_dev = []

Y_dev = []

for i, entry in tqdm(df_dev.iterrows(), total=len(df_dev)):
    if args.video_scores_path!='/home/semantic/media-memorability/textual_scores/data_2021/Memento10k Data/training_set/train_scores.csv':
        y = (entry['short_term'], entry['short_term_norm'],entry['long_term'])
    else:
        y = (entry['short_term'],entry['short_term_norm'])
    x = [d['vector'] for d in features if d['label']==entry['video_id']]
    X_dev.append(x)
    Y_dev.append(y)

X_dev = np.array(X)
Y_dev = np.array(Y)
        
     

                        
                        
X_test=[]                        

for i, entry in tqdm(df_test.iterrows(), total=len(df_test)):
    if args.video_scores_path!='/home/semantic/media-memorability/textual_scores/data_2021/Memento10k Data/training_set/train_scores.csv':
        y = (entry['short_term'], entry['short_term_norm'],entry['long_term'])
    else:
        y = (entry['short_term'],entry['short_term_norm'])
    x = [d['vector'] for d in features if d['label']==entry['video_id']]
    X_test.append(x)


X_test = np.array(X)

        









X_train = X['X']
y_train = Y_st
svr_model = SVR(C=1e-05, gamma='scale', kernel='linear')
svr_model.fit(X_train, y_train)
#ismail_st_test = svr_model.predict(X_test_w2v)
ismail_st_dev = svr_model.predict(X_dev_bert3)


ismail_st_test = svr_model.predict(X_test_bert3)


X_train = X['bert3']
y_train = Y_lt
svr_model_lt = SVR(C=1e-05, gamma='scale', kernel='linear')
svr_model_lt.fit(X_train, y_train)
ismail_lt_test = svr_model_lt.predict(X_test_w2v)


ismail_preds = pd.DataFrame({'video_id': df_text_test.video_id.unique(),
                             'short_term': ismail_st_test,
                             'long_term': ismail_lt_test})

ismail_preds.to_csv('me_2020/ismail_testset_preds_svr_1e-05_scale_lin.csv')
# ismail_preds = pd.read_csv('me_2020/ismail_testset_preds_svr_1e-05_scale_lin.csv')"""
