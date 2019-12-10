import json
import plotly
import pandas as pd
import sys
import re
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

def tokenize(text):
    '''
    INPUT:
        TEXT (string) : text to tokenize/lemmatize
    OUTPUT:
        CLEAN_WORDS (list) : list of tokenized/cleaned words
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

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


# load data
engine = create_engine('sqlite:///messages.db')
df =pd.read_sql("SELECT * FROM messages", engine)

# load model
model = joblib.load("classifier.pkl")

X = df.message.values
y = df.drop(['id', 'message', 'original','genre'], axis=1).values




# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_counts = df.iloc[:,5:].sum()
    category_names = list(category_counts.index)
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }, 
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    
    query = pd.DataFrame([query], columns=['message'])
    # use model to predict classification for query
    classification_labels = model.predict(query.values[0])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
     
    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query.message.values[0],
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()