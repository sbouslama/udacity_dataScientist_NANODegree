 # import libraries
import pandas as pd
import numpy as np
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
sys.path.append('./customClasses')
from utils import *

def build_model(X_train,y_train):
    '''
    INPUT 
        X_Train: Training features for use by GridSearchCV
        y_train: Training labels for use by GridSearchCV
    OUTPUT
        Returns a pipeline model that has gone through cleaning, word embedding using doc2vec, 
        and created into a ML model
    '''
    # pipeline = Pipeline(steps = [
    #    ('processing', TextProcessor()),
    #    ('doc2vec',doc2vec_transform()),
    #    ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators=10)))
    # ],
    # )

    pipeline = Pipeline(steps = [
       ('processing', TextProcessor()),
       ('count vect', CountVectorizer()),
       ('tfidf',TfidfTransformer()),
       ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators=10)))
    ],
    )
    parameters = {  
        'clf__estimator__min_samples_split': [2]
        }
#     parameters = {  
#         'clf__estimator__min_samples_split': [2, 4],
#         'clf__estimator__max_features': ['log2', 'auto', 'sqrt', None],
# #         'clf.__estimator__n_estimators':[10,20,30],

#     }
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    cv.fit(X_train,y_train)
    print('best parameters are:', cv.best_params_)
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
        df = pd.read_sql("SELECT * FROM {}".format(conn.table_names()[0]), conn)

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

        # print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py /data_path_to_disaster_dataset.db classifier.pkl')


if __name__ == '__main__':
    main()
