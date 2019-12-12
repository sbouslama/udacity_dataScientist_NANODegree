# Project Description

The U.S. has almost 500 students for every guidance counselor. Underserved youth lack the network to find their career role models, making CareerVillage.org the only option for millions of young people in America and around the globe with nowhere else to turn.

To date, 25,000 volunteers have created profiles and opted in to receive emails when a career question is a good fit for them. This is where your skills come in. To help students get the advice they need, the team at CareerVillage.org needs to be able to send the right questions to the right volunteers. The notifications sent to volunteers seem to have the greatest impact on how many questions are answered.

The objective: develop a method to recommend relevant questions to the professionals who are most likely to answer them.

# Try it

## install requirements
To install requirements, please use this command line. 
`pip install -r requirements.txt`
In order to run it you will also need to download nltk stopwords. So in python interpreted run:
`import nltk
 nltk.download('stopwords')
` 

# Project Details

- Data: this folder contains csv files from competiton
- utils: this folder contains a packaged useful functions
- Model: this forlder cotains the model implemented 
- Preprocessors: this folder contains all the features preprocessors for both the question and students entities. 
- Recommender: this folder contains the recommendation engine developed for the compagany 
