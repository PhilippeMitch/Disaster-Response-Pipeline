import re
import sys
import nltk
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer that extracted whether each text started with a verb
    
    VB: verb
    VBP: present tense verb
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def load_data(database_filepath):
    """
    Load Data Function
    
    Input:
    -----
        database_filepath: str
            path to SQLite db
    Output:
    ------
        X: Dataframe
            feature DataFrame
        y: Dataframe
            label DataFrame
        category_names: List
            List of name's category
    """
    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table(database_filepath.split('.')[0] + '_table',engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    splitting a sentence into a sequence of words.
    
    input:
    ------
        text: str
    output
    ------
        clean_tokens: List
    """
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
     # tokenize text
    tokens = word_tokenize(text)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """
    Build Model function
    
   This machine pipeline take in the message column as input and output 
   classification results on the other 36 categories in the dataset. 
    """
    # Create a pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),
    
         ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to evaluate the model
    
    Report the f1 score, precision and recall for each output category of the dataset. 
        
    Inputs:
        y_true: numpy.ndarray
            dataset labels
        y_pred: numpy.ndarray
            model predictions
    
    Output:
        f1score -> customized fscore
    """
    y_pred = model.predict(X_test)
    # Print classification report on test data
    for i in range(y_pred.shape[1]):
        print("=======================",Y_test.columns[i],"======================")
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))
        
    accuracy = (y_pred == Y_test).mean().mean()
    print('Average overall accuracy {0:.2f}% \n'.format(accuracy*100))

def save_model(model, model_filepath):
    pickle.dump(model, open("models/"+model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()