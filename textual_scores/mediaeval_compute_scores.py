#!/usr/bin/env python
# coding: utf-8

import pickle
import string
import argparse 
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 


parser = argparse.ArgumentParser(description='Computing text scores for MediaEval 2020')

parser.add_argument("-d", "--deep_caption_path", type=str, help="Path to the file containing deep captions")
parser.add_argument("-wv", "--word_embeddings_path", type=str, help="Path to word embeddings (e.g. GloVe)", default=None)
parser.add_argument('-s', '--save_path', type=str, help="Path to save the Short and Long predictions")

args = parser.parse_args()


stopwords = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", 
             "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", 
             "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", 
             "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", 
             "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", 
             "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", 
             "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", 
             "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", 
             "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", 
             "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", 
             "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", 
             "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to",
             "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren",
             "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won",
             "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", 
             "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", 
             "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", 
             "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", 
             "would"]

lemmatizer = WordNetLemmatizer()

def tokenize(s):
    numbers = {'2': 'two', '3': 'three', '4': 'four'}
    s = ''.join(c for c in s if c not in string.punctuation or c == ' ').lower()
    t = RegexpTokenizer(r'\w+').tokenize(s)
    t = [lemmatizer.lemmatize(w) if w not in numbers else numbers[w] for w in t if w not in stopwords]
    return ' '.join(t)

data = pd.read_csv(args.deep_caption_path)

if args.word_embeddings_path:
    w2v = KeyedVectors.load_word2vec_format(args.word_embeddings_path)

else:
    w2v = pickle.load(open('../conceptnet/glove.6B/glove.w2v.6B.300d.pickle', 'rb'))

model_st = pickle.load(open('me20_svr_w2v_st_model.pickle', 'rb'))
model_lt = pickle.load(open('me20_svr_w2v_lt_model.pickle', 'rb'))

y_st_pred = model_st.predict(X_w2v)
y_lt_pred = model_lt.predict(X_w2v)

data['results_st'] = y_st_pred
data['results_lt'] = y_lt_pred

data[['caption', 'video_id', 'video', 'frame', 'results_st', 'results_lt']].to_csv(args.save_path)
print('Results saved into', args.save_path)