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
from functools import reduce



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


#df_data = pd.read_csv('/home/semantic/media-memorability/textual_scores/data_2021/TRECVid Data/training_set/train_scores.csv')
df_data=pd.read_csv('textual_scores/data_with_perplexity.csv')
print(df_data.columns)               
                        

    
      
#Y_st=np.array(Y_st).reshape(-1, 1) 
                    
# video results
#video_scores=pd.read_csv('PicSOM_prediction/visual_pred_training_set_trecvid.csv')
X_st_video=pd.read_csv('PicSOM_prediction/audio_and_visual_predictions_2021/trecvid_short_260_i3d-25-128-avg--600.csv',header=None,names=["video_id", "score_st_video"])
X_st_norm_video=pd.read_csv('PicSOM_prediction/audio_and_visual_predictions_2021/trecvid_norm_280_i3d-25-128-avg--660.csv',header=None,names=["video_id", "score_norm_video"])
X_lt_video=pd.read_csv('PicSOM_prediction/audio_and_visual_predictions_2021/trecvid_long_130_i3d-25-128-avg--680.csv',header=None,names=["video_id", "score_lt_video"])

     
# text's results
text_scores=pd.read_csv('textual_scores/text_pred_training_set_trecvid.csv')



   
#X_st=np.c_[video_scores.short_term_pred,text_scores.short_term_pred]
#X_st_norm=np.c_[video_scores.short_term_norm_pred,text_scores.short_term_norm_pred]
#X_lt=np.c_[video_scores.long_term_pred,text_scores.long_term_pred]


X_st_text=pd.read_csv('textual_scores/trecvid_short_160_bert3--460.csv',header=None,names=["video_id", "score_st_text"])
#print(X_st_text['number'])
X_st_norm_text=pd.read_csv('textual_scores/trecvid_norm_140_bert3--460.csv',header=None,names=["video_id", "score_norm_text"])
X_lt_text=pd.read_csv('textual_scores/trecvid_long_90_bert3--720.csv',header=None,names=["video_id", "score_lt_text"])



frames=[df_data,X_st_video,X_lt_video,X_st_norm_video,X_st_text,X_st_norm_text,X_lt_text]
data_frames = reduce(lambda  left,right: pd.merge(left,right,on=['video_id'],
                                            how='outer'), frames)

print(data_frames)

X_st=np.c_[data_frames.score_st_text,data_frames.score_st_video,data_frames.probability]
X_st_norm=np.c_[data_frames.score_norm_text,data_frames.score_norm_video,data_frames.probability]
X_lt=np.c_[data_frames.score_lt_text,data_frames.score_lt_video,data_frames.probability]







#X_st=np.c_[X_st , perplexity['probability']]

#X_st=perplexity['probability']
#X_st=np.array(X_st).reshape(-1, 1)


 # Ground Truth
Y_st = df_data['short_term']#.values#.tolist()
Y_st_norm= df_data['normalized_short_term']#.values#.tolist()
if dataset=='trecvid':
    Y_lt = df_data['long_term']#.values#.tolist()
    
    
    
    
    

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

X = {'ST_norm': X_st_norm,'LT': X_lt}


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



# # Combining everything

# In[51]:



                                                      
#print(df_data)                                                         
                                                           
df_data.to_csv('text_pred_training_set_trecvid.csv')  

## Testset and dev results
df_text_test = pd.read_csv(args.test_set_path)
df_text_dev = pd.read_csv(args.dev_set_path)




if args.test_set_path=='data_2021/Memento10k Data/test_set/test_text_descriptions.csv':
                        df_text_test['description']=df_text_test['description_0']+df_text_test['description_1']+df_text_test['description_2']+df_text_test['description_3']+df_text_test['description_4']
        
if args.dev_set_path=='data_2021/Memento10k Data/dev_set/dev_text_descriptions.csv':
                        df_text_dev['description']=df_text_dev['description_0']+df_text_dev['description_1']+df_text_dev['description_2']+df_text_dev['description_3']+df_text_dev['description_4']
        
        
        
        


test_text_concat = df_text_test[['video_id','description']].groupby(['video_id'])['description'].transform(lambda x: '  '.join(x)).drop_duplicates()

dev_text_concat = df_text_dev[['video_id','description']].groupby(['video_id'])['description'].transform(lambda x: '  '.join(x)).drop_duplicates()


df_test = pd.DataFrame({"video_id": df_text_test.video_id.unique(),
                        "text": test_text_concat,})


df_dev = pd.DataFrame({"video_id": df_text_dev.video_id.unique(),
                        "text": dev_text_concat,})




corpus_test = df_test.text.values
corpus_test_tokenized2 = [tokenize2(s) for s in corpus_test]
#sbert3_test=SentenceTransformer('fabriceyhc/bert-base-uncased-yahoo_answers_topics')
X_test_bert3_embeddings = sbert3.encode(corpus_test_tokenized2)
with open('Bert3_embeddings_test_memento.pkl','wb') as f:
    pickle.dump(X_test_bert3_embeddings, f)


corpus_dev = df_dev.text.values
corpus_dev_tokenized2 = [tokenize2(s) for s in corpus_dev]
#sbert3_test=SentenceTransformer('fabriceyhc/bert-base-uncased-yahoo_answers_topics')
X_dev_bert3_embeddings = sbert3.encode(corpus_dev_tokenized2)
with open('Bert3_embeddings_dev_memento.pkl','wb') as f:
    pickle.dump(X_dev_bert3_embeddings, f)





X_test_tfidf = []
X_test_w2v   = []

for i, entry in tqdm(df_test.iterrows(), total=len(df_test)):
    text = tokenize(entry['text'])
    
    x_tfidf = vectorizer.transform([text]).toarray()[0]
    words = [word for word in text.split(' ') if word in w2v]
    x_w2v = np.zeros([300]) if not words else np.mean([w2v[word] for word in words], axis=0)
    
    X_test_tfidf.append(x_tfidf)
    X_test_w2v.append(x_w2v)
    
    


X_train = X['bert3']
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
# ismail_preds = pd.read_csv('me_2020/ismail_testset_preds_svr_1e-05_scale_lin.csv')
