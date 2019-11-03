**requirements**
to install all the requirements needed in this project you should executes this command:
pip install -r requirements.txt

**Project details**
- etl_pipeline.py:

Python script including the data pipeline for the dataset.
It uses the ETL pipeline which is the Extact, Transforl and Load process.
to run it, you can use the line command:

python etl_pipeline.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

- train_classifier.py:

Python script including the machine learning portion. In this pipeline, we will
go through the following steps:

1- spliting the data into a training and testing set

2- creating the pipeline with scikit-learn's Pipeline and GridSearchCV
to output a final model that uses the message column to predict classifications for 36
categories (multi-output classification).

3- exporting the model to a pickle file.

- App:
Results will be displayed in a flask web app. you can type this command to 
run the web app:
python run.py
