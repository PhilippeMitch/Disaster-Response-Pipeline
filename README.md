# Disaster Response Pipeline Project
In this project, you'll apply these skills to analyze disaster data from [Appen] (https://www.figure-eight.com/)(opens in a new tab) (formerly Figure 8) to build a model for an API that classifies disaster messages.It includes a web app where an emergency worker can input a new message and get classification results in several categories.

Project Components
There are three components you'll need to complete for this project.

### 1. ETL Pipeline
In a Python script, process_data.py, write a data cleaning pipeline that:

* Loads the messages and categories datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database
  
### 2. ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that:

* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/YourDatabaseName.db`
      
      ![](https://github.com/PhilippeMitch/Disaster-Response-Pipeline/blob/main/images/data_preprocess_local.png)
      
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/YourDatabaseName.db classifier.pkl`

      ![](https://github.com/PhilippeMitch/Disaster-Response-Pipeline/blob/main/images/train_models_local.png)

      ![](https://github.com/PhilippeMitch/Disaster-Response-Pipeline/blob/main/images/train_models_local_1.png)

2. Go to `app` directory: `cd app`

4. Run your web app: `python run.py data/YourDatabaseName.db`

![](https://github.com/PhilippeMitch/Disaster-Response-Pipeline/blob/main/images/run_app_local.png)

5. Click the `PREVIEW` button to open the homepage
