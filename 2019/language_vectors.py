import pandas as pd
from bert_serving.client import BertClient
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import *
import json
import numpy as np


def read_titles(path):
    """Load the captions into a dataframe"""
    stemmer = PorterStemmer()
    vn = []
    cap = []
    df = pd.DataFrame()
    with open(path) as f:
        for line in f:
            pairs = line.split()
            vn.append(pairs[0])
            #singles = [stemmer.stem(pairs[1])]
            cap.append(pairs[1])
        df['video']=vn
        df[['video', 'poubelle']] = df['video'].str.split(".webm", expand=True, )
        df[['poubelle2', 'video']] = df['video'].str.split("video", expand=True, )
        df['caption']=cap
    return df

def Bert_format (path):
# Creating train and dev dataframes according to BERT
    """Load the captions into a dataframe"""
    vn = []
    cap = []
    df = pd.DataFrame()
    with open(path) as f:
        for line in f:
            pairs = line.split()
            vn.append(pairs[0])
            cap.append(pairs[1])
        df['video'] = vn
        df['caption'] = cap
        df['alpha'] =['a']*len(cap)
    return df


def read_alto_captions(path):
    df = pd.read_csv(path, names=['caption'])
    df[['videos','caption_alto']] = df.caption.str.split("A",expand=True,)
    df[['trash','video']]=df.videos.str.split("          ",expand=True,)
    df['video']=df['video'].str.strip("0")
    df[['video','poubelle']] =df['video'].str.split(":83",expand=True,)
    df['caption_alto'].fillna('a',inplace=True)
    #print(df['video'])
    return df[['video','caption_alto']]

def alto_and_titles(alto,titles):
    #print(titles)
    #df = pd.concat([alto, titles], axis=1, join='inner')
    #print(alto)
    #print(titles)
    df=pd.merge(alto, titles, on='video')
    #print(df)
    #print(df['videos'])
    df2 = pd.DataFrame()
    df2['caption']= df[['caption_alto', 'caption']].apply(lambda x: ' '.join(x), axis=1)
    df2['video']=df['video']
    print(df2['caption'][3])
    return df2


def adding_danny_to_rest(alto_and_titles,danny_captions):
    df=pd.merge(danny_captions,alto_and_titles, on="video")
    #print(df['videos'])
    df2 = pd.DataFrame()
    df2['caption']= df[['caption', 'captions_danny']].apply(lambda x: ''.join(x), axis=1)
    df2['video'] = df['video']
    print(df2['caption'][3])
    return df2

def getting_Bertclient_vectors(column):
    bc = BertClient()
    vectors= bc.encode(column.tolist())
    #print(vectors)
    df_vec = pd.DataFrame(vectors)
    df_vec.to_csv("bert_alto.csv")
    return df_vec


def obtaining_BOW_vectors(df):
    #cv = CountVectorizer(ngram_range=(2,4),stop_words='english',max_features=20000)
    cv = CountVectorizer(stop_words='english')
    X_CV = cv.fit_transform(df['caption'])
    X= X_CV.toarray()
    return X







def obtaining_tfidf_vectors(df):
    cv = TfidfVectorizer()
    X_CV = cv.fit_transform(df['caption'])
    X= X_CV.toarray()
    return X
