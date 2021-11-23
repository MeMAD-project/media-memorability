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


dataset='trecvid'
#dataset='memento'


df_data = pd.read_csv('/home/semantic/media-memorability/textual_scores/data_2021/TRECVid Data/training_set/train_scores.csv')

                        
                        
 # Ground Truth
Y_st = df_data['scores_raw_short_term']#.values#.tolist()
Y_st_norm= df_data['scores_normalized_short_term']#.values#.tolist()
if dataset=='trecvid':
    Y_lt = df_data['scores_raw_long_term']#.values#.tolist()
    
    
    
#Y_st=np.array(Y_st).reshape(-1, 1) 
                    
# video results
video_scores=pd.read_csv('PicSOM_prediction/visual_pred_training_set_trecvid.csv')

       
# text's results
text_scores=pd.read_csv('textual_scores/text_pred_training_set_trecvid.csv')


   
X_st=np.c_[video_scores.short_term_pred,text_scores.short_term_pred]
X_st_norm=np.c_[video_scores.short_term_norm_pred,text_scores.short_term_norm_pred]
X_lt=np.c_[video_scores.long_term_pred,text_scores.long_term_pred]



perplexity=pd.read_csv('textual_scores/data_with_perplexity.csv')




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





df_data['short_term_pred_ens']=0
folds = {}
print('Short term memorability prediction:'.upper())

X = {'ST': X_st,'LT': X_lt}

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
                df_data.loc[df_data.index== test_index[j], 'short_term_pred_ens']=y_pred[j]
            folds[k][model_name].append((y_pred, y_test))
            print(f'done! ({(time.time() - t):.2} secs). Spearman: {spearman(y_pred, y_test):.2}')
            
            t = time.time()
            
df_data['short_term_norm_pred_ens']=0         

folds_st_norm = {}
print('Short term norm memorability prediction:'.upper())

X = {'ST_norm': X_st_norm}

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
                df_data.loc[df_data.index== test_index[j], 'short_term_norm_pred_ens']=y_pred[j]
            folds_st_norm [k][model_name].append((y_pred, y_test))
            print(f'done! ({(time.time() - t):.2} secs). Spearman: {spearman(y_pred, y_test):.2}')
            
            t = time.time()
            


df_data['long_term_pred_ens']=0 
folds_lt = {}
print('Long term memorability prediction:'.upper())

X = {'ST': X_st,'LT': X_lt}


if dataset=='trecvid':
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
                    df_data.loc[df_data.index== test_index[j], 'long_term_pred_ens']=y_pred[j]
                folds_lt[k][model_name].append((y_pred, y_test))
                print(f'done! ({(time.time() - t):.2} secs). Spearman: {spearman(y_pred, y_test):.2}')

                t = time.time()

                
                
if dataset=='trecvid':                                                         
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
            else:
                sps = [spearman(y_p, y_t) for y_p, y_t in all_folds[embedding][model_name]]
            print(round(sum(sps)/len(sps), 4))

            
            
  

#ismail_st = folds['w2v']["SVR(C=1e-05, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='scale',\n    kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)"]
#ismail_lt = lt_folds['w2v']["SVR(C=1e-05, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='scale',\n    kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)"]


#X_train = X['bert3']              
svr_model = SVR(C=1e-05, gamma='scale', kernel='linear')
svr_model.fit(X_train, y_train)
#ismail_st_test = svr_model.predict(X_test_w2v)
#ismail_st_dev = svr_model.predict(X_dev_bert3)


st_test = svr_model.predict(X_test)



y_train = Y_lt
svr_model_lt = SVR(C=1e-05, gamma='scale', kernel='linear')
svr_model_lt.fit(X_train, y_train)
lt_test = svr_model_lt.predict(X_test)


preds = pd.DataFrame({'video_id': df_text_test.video_id.unique(),
                             'short_term': st_test,
                             'long_term': lt_test})

ismail_preds.to_csv('me_2020/ismail_testset_preds_svr_1e-05_scale_lin.csv')
# ismail_preds = pd.read_csv('me_2020/ismail_testset_preds_svr_1e-05_scale_lin.csv')
