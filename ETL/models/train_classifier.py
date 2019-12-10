# import libraries
import sys
import pickle
import re

import os
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sqlalchemy import create_engine
import warnings
from contextlib import redirect_stdout
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    """
    INPUT:
        database_filepath (string) : database location
    OUTPUT:
        X (np.array) : messages to process
        y (np.array) : training/evaluating categories
        labels (np.array) : list of message classification labels
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM {}".format(engine.table_names()[0]), engine)
    X = df.message.values
    y = df.drop(['id', 'message', 'original','genre'], axis=1).values
    labels = (df.drop(['id', 'message', 'original','genre'],
                      axis=1)).columns.values
    return X, y, labels


def tokenize(text):
    '''
    INPUT:
        TEXT (string) : text to tokenize/lemmatize
    OUTPUT:
        CLEAN_WORDS (list) : list of tokenized/cleaned words
    '''

    detected_urls = re.findall(url_regex, text)  # find urls
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')  # replace urls

    tokens = word_tokenize(
        text)  # tokenizer object, not capitalised as it is a class method

    words = [word for word in tokens if word not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()  # parent class lemmatizer object

    clean_words = []  # empty list for results

    for word in words:
        clean_word = lemmatizer.lemmatize(
            word).lower().strip()  # return lemmatized words

        clean_words.append(clean_word)  # append cleaned/lemmatized string

    return clean_words



def build_model():
    pipeline = Pipeline(
        [('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),        
        ])),

         ('clf', MultiOutputClassifier(RandomForestClassifier()))
         ])

    parameters = {'clf__estimator__min_samples_split': [2, 3]}
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, y_test, labels):
    """
    Evaluates a pre-trained classifier
    Returns accuracy, recall, and precision for each unique label
    """
    preds = model.predict(X_test)
    for label in range(0, len(labels)):
        print('Message category:', labels[label])
        print(classification_report(y_test[:, label], preds[:, label]))


def save_model(model, model_filepath):
    """
    Save model to a pickle file
    """
    joblib.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, labels = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)
        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, labels)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
