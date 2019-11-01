import pandas as pd
from sqlalchemy import create_engine
import re
import random
import sys
import pickle
from sklearn.pipeline import Pipeline
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


class TextProcessor( BaseEstimator, TransformerMixin ):
    """
    Class for carrying all the text pre-processing stuff throughout the project
    """

    def __init__(self):
        self.stopwords = stopwords.words('english')
        self.ps = PorterStemmer()

        # stemmer will be used for each unique word once
        self.stemmed = dict()

    def process(self, text: str, allow_stopwords: bool = False) -> str:
        """
        Process the specified text,
        splitting by non-alphabetic symbols, casting to lower case,
        removing stopwords, HTML tags and stemming each word

        :param text: text to process
        :param allow_stopwords: whether to remove stopwords
        :return: processed text
        """
        ret = []
        url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        detected_urls = re.findall(url_regex, text)
        for url in detected_urls:
            text = text.replace(url, "urlplaceholder")
        # split and cast to lower case
        text = re.sub(r'<[^>]+>', ' ', str(text))
        for word in re.split('[^a-zA-Z]', str(text).lower()):
            # remove non-alphabetic and stop words
            if (word.isalpha() and word not in self.stopwords) or allow_stopwords:
                if word not in self.stemmed:
                    self.stemmed[word] = self.ps.stem(word)
                # use stemmed version of word
                ret.append(self.stemmed[word])
        return ' '.join(ret)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X['message'] = X['message'].apply(lambda x: self.process(x, allow_stopwords=True))
        return X

class doc2vec_transform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def train_d2v(self, df: pd.DataFrame, dim: int) -> (Doc2Vec, dict):
        """
        Train Doc2Vec object on provided data
        :param df: data to work with
        :param target: column name of target entity in df to train embeddings for
        :param features: list of feature names to be used for training
        :param dim: dimension of embedding vectors to train
        :return: trained Doc2Vec object
        """
        prepared = [TaggedDocument(s.split(), [str(i)]) for i, s in enumerate(df['message'])]

        d2v = Doc2Vec(prepared, vector_size=dim, workers=4, epochs=10, dm=0)
        docvecs = {d2v.docvecs.index2entity[i]: d2v.docvecs.vectors_docs[i]
                   for i in range(len(d2v.docvecs.index2entity))}
        return d2v, docvecs


    def pipeline_d2v(self, df: pd.DataFrame, dim: int):
        """
        Pipeline for training embeddings for messages via doc2vec algorithm

        :param df: raw df.csv dataset
        :param dim: dimension of doc2vec embeddings to train
        :return: trained msgs and Doc2Vec model
        """

        # train and save message embeddings
        _, msg_embs = self.train_d2v(df, dim)
        msg_embs = pd.DataFrame(msg_embs).transpose()
        return msg_embs
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        emb = self.pipeline_d2v(X, dim=10)
        return emb

