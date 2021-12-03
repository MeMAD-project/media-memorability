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

#from transformers import AutoTokenizer, AutoModelForSequenceClassification
  
#tokenizer = AutoTokenizer.from_pretrained("fabriceyhc/bert-base-uncased-yahoo_answers_topics")

#model = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-yahoo_answers_topics")



parser = argparse.ArgumentParser(description='Computing text scores for MediaEval 2021')

parser.add_argument("-d", "--video_descriptions_path", type=str, help="Path to the CSV file containing video IDs and corresponding description(s)", default='data_2021/TRECVid Data/training_set/train_text_descriptions.csv')
#parser.add_argument("-d", "--video_descriptions_path", type=str, help="Path to the CSV file containing video IDs and corresponding description(s)", default='data_2021/Memento10k Data/training_set/train_text_descriptions.csv')

parser.add_argument("-s", "--video_scores_path", type=str, help="Path to the CSV file containing ground-truth scores.", default="data_2021/TRECVid Data/training_set/train_scores.csv")
#parser.add_argument("-s", "--video_scores_path", type=str, help="Path to the CSV file containing ground-truth scores.", default="data_2021/Memento10k Data/training_set/train_scores.csv")



#parser.add_argument("-dev", "--dev_set_path", type=str, help="Path to the CSV file containing video IDs and corresponding description(s)", default="data_2021/TRECVid Data/dev_set/dev_text_descriptions.csv")

parser.add_argument("-dev", "--dev_set_path", type=str, help="Path to the CSV file containing video IDs and corresponding description(s)", default="data_2021/Memento10k Data/dev_set/dev_text_descriptions.csv")


#parser.add_argument("-devscores", "--dev_scores_path", type=str, help="Path to the CSV file containing ground-truth scores dev set.", default="data_2021/TRECVid Data/training_set/dev_scores.csv")
                    
parser.add_argument("-devscores", "--dev_scores_path", type=str, help="Path to the CSV file containing ground-truth scores dev set.", default="data_2021/TRECVid Data/Memento10k Data/dev_set/dev_scores.csv")




#parser.add_argument("-t", "--test_set_path", type=str, help="Path to the CSV file containing video s of the testset.", default="data_2021/TRECVid Data/test_set/test_text_descriptions.csv")

parser.add_argument("-t", "--test_set_path", type=str, help="Path to the CSV file containing video s of the testset.", default="data_2021/Memento10k Data/test_set/test_text_descriptions.csv")



                    
parser.add_argument("-r", "--results_path", type=str, help="Path to where to save the results for short and long term predictions.", default="data_2021/test_text_descriptions.csv")
parser.add_argument("-wv", "--word_embeddings_path", type=str, help="Path to word embeddings (e.g. GloVe)", default=None)
parser.add_argument('--save_model', action='store_true', default=False)

args = parser.parse_args()


# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()
spearman = lambda x,y: spearmanr(x, y).correlation

if args.word_embeddings_path:
    w2v = KeyedVectors.load_word2vec_format(args.word_embeddings_path)

else:
    w2v = pickle.load(open('/data/glove.6B/glove.w2v.6B.300d.pickle', 'rb'))


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
                    
   
                    
#For memento dataset
if args.video_descriptions_path=='data_2021/Memento10k Data/training_set/train_text_descriptions.csv':
                        df_text['description']=df_text['description_0']+df_text['description_1']+df_text['description_2']+df_text['description_3']+df_text['description_4']
#df_text['scores_raw_long_term']=np.random.randint(1, 6, df_text.shape[0])
        


# For videos having multiple descriptions
text_concat = df_text[['video_id','description']].groupby(['video_id'])['description'].transform(lambda x: ' '.join(x)).drop_duplicates()

#deep_captions = [ l[8:].strip() for l in open(args.deep_caption_path)]

df_data = df_scores.copy()
df_data['text'] = text_concat.values
                    
df_data['short_term'] = df_data['scores_raw_short_term']
df_data['short_term_norm'] = df_data['scores_normalized_short_term']

if args.video_descriptions_path!='data_2021/Memento10k Data/training_set/train_text_descriptions.csv':
    df_data['long_term'] = df_data['scores_raw_long_term']
    df_data = df_data[['video_id', 'text', 'short_term', 'long_term','short_term_norm']]
else:
    df_data = df_data[['video_id', 'text', 'short_term','short_term_norm']]
#df_data['deep_caption'] = deep_captions[:590]



df_data['content'] = df_data['text'] #+ '  ' + df_data['deep_caption']


corpus = df_data.text.values

corpus_tokenized = [tokenize3(s) for s in corpus]
corpus_tokenized2 = [tokenize2(s) for s in corpus]

corpus_tokenized3 = [tokenize3(s) for s in corpus]

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
    if args.video_descriptions_path!='data_2021/Memento10k Data/training_set/train_text_descriptions.csv':
        y = (entry['short_term'],entry['short_term_norm'], entry['long_term'])
    else:
        y = (entry['short_term'],entry['short_term_norm'])
        
    
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


#model = 

sbert1 = SentenceTransformer('distiluse-base-multilingual-cased')
sbert2 = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
sbert3=SentenceTransformer('fabriceyhc/bert-base-uncased-yahoo_answers_topics')

#sbert3=SentenceTransformer('bhadresh-savani/distilbert-base-uncased-emotion')
#sbert3=SentenceTransformer('facebook/bart-large-xsum')
#sbert3=SentenceTransformer('microsoft/SportsBERT')
#sbert3=SentenceTransformer("t5-base")
#sbert3=SentenceTransformer("deepset/roberta-base-squad2")

#sbert3=SentenceTransformer("joeddav/bart-large-mnli-yahoo-answers")
#sbert3=SentenceTransformer("miesnerjacob/youtube-garm-bart-large-11-03-2021")

"""     with torch.no_grad():
        outputs = sbert3(corpus_tokenized2, labels=corpus_tokenized2.clone())
        neg_log_likelihood = outputs[0] * trg_len"""

        
        
        
        
bert1_embeddings = sbert1.encode(corpus_tokenized2)
bert2_embeddings = sbert2.encode(corpus_tokenized2)
bert3_embeddings = sbert3.encode(corpus_tokenized2)
print(bert3_embeddings.shape)

#with open('Bert3_embeddings_training_memento.pkl','wb') as f:
    #pickle.dump(bert3_embeddings, f)





#perplexity=pd.read_csv('data_with_perplexity.csv')




#bert3_embeddings=np.c_[bert3_embeddings , perplexity['probability']]


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


X = {'bert3': bert3_embeddings}#,'tfidf': X_tfidf, 'w2v':X_w2v, 'bert1': bert1_embeddings, 'bert2': bert2_embeddings}#,'bert3': bert3_embeddings}
if args.video_descriptions_path!='data_2021/Memento10k Data/training_set/train_text_descriptions.csv':
    Y_st = Y[:, 0]
    Y_st_norm = Y[:, 1]
    Y_lt = Y[:, 2]
else:
    Y_st=Y[:, 0]
    Y_st_norm = Y[:, 1]
    

df_data['short_term_pred']=0
folds = {}
print('Short term memorability prediction:'.upper())

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
if args.video_descriptions_path!='data_2021/Memento10k Data/training_set/train_text_descriptions.csv':
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
