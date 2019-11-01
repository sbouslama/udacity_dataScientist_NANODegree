# import libraries
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


def build_model(X_train,y_train):
    '''
    INPUT 
        X_Train: Training features for use by GridSearchCV
        y_train: Training labels for use by GridSearchCV
    OUTPUT
        Returns a pipeline model that has gone through cleaning, word embedding using doc2vec, 
        and created into a ML model
    '''
    pipeline = Pipeline(steps = [
       ('processing', TextProcessor()),
       ('doc2vec',doc2vec_transform()),
       ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators=10)))
    ],
    )
    
    parameters = {  
        'clf__estimator__min_samples_split': [2, 4],
        'clf__estimator__max_features': ['log2', 'auto', 'sqrt', None],
#         'clf.__estimator__n_estimators':[10,20,30],

    }
    
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    cv.fit(X_train,y_train)
    return cv

def evaluate_model(pipeline, X_test, Y_test):
    '''
    INPUT 
        pipeline: The model that is to be evaluated
        X_test: Input features, testing set
        y_test: Label features, testing set
        category_names: List of the categories 
    OUTPUT
        This method does nto specifically return any data to its calling method.
        However, it prints out the precision, recall and f1-score
    '''
    # predict on test data
    pred = pipeline.predict(X_test)
    for i in range(0,Y_test.shape[1]):
        print(Y_test.keys()[i])
        print(classification_report(Y_test.iloc[:,i], pred[:,i]))


def save_model(model, model_filepath):
    '''
    Saves the model to disk
    INPUT 
        model: The model to be saved
        model_filepath: Filepath for where the model is to be saved
    OUTPUT
        While there is no specific item that is returned to its calling method, this method will save the model as a pickle file.
    '''    
    temp_pickle = open(model_filepath, 'wb')
    pickle.dump(model, temp_pickle)
    temp_pickle.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        conn = create_engine('sqlite:///{}'.format(database_filepath))
        # *todo* make it generalised
        df = pd.read_sql("SELECT * FROM disaster_dataset", conn)
        X = df.iloc[:,1:2]
        Y = df.iloc[:,4:]
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train,Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py /data_path_to_disaster_dataset.db classifier.pkl')


if __name__ == '__main__':
    main()
