import re
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from gensim.sklearn_api import D2VTransformer
from gensim.parsing.preprocessing import preprocess_string



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
    
    def transform(self, X, column):
        X[column] = X[column].apply(lambda x: self.process(x, allow_stopwords=True))
        return X

class doc2vec_transform(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = None
    
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


    
    def fit(self, X, y=None):
        self.model, msg_embs = self.train_d2v(X, dim=10)
        return self
    
    def transform(self, X):
        if self.model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        return np.asmatrix(
            np.array(
                [self.model.infer_vector(preprocess_string(row['message'])
                    ) for index, row in X.iterrows()]))
