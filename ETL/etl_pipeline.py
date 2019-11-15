# import libraries
import pandas as pd
import sys
from sqlalchemy import create_engine

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [col.split('-')[0] for col in row]
    print('There are {} categories with the following names: {}'.format(len(category_colnames), category_colnames))
    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:x.split('-')[1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df=pd.concat([df, categories], axis=1)
    # check number of duplicates
    print('there is {} duplicated rows'.format(df.duplicated().sum()))
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, name='messages' ):
    engine = create_engine('sqlite:///{}.db'.format(name))
    df.to_sql(name, engine, index=False)


def main():
    
    if sys.argv:
        # load messages and categories datasets
        messages_path = sys.argv[1]
        categories_path = sys.argv[2]
        messages = pd.read_csv(messages_path)
        categories = pd.read_csv(categories_path)
        # merge datasets
        df = messages.merge(categories, on='id', how='left')
        print('CLEANING DATA ...')
        df = clean_data(df)
        # save the clean data
        print('SAVING DATA ...')
        save_data(df)
        
    else: 
        print('Please provide the filepath of the disaster messages and categories datasets '\
              'as arguments. \n\nExample: python '\
              'etl_pipeline.py ./messages.csv ./categories.csv')

main()