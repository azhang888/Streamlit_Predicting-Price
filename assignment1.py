# import statements 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.pipeline import Pipeline
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Filename of the dataset to use for training and validation
train_data = "Mobile Price_train.csv"
# Filename of test dataset to apply your model and predict outcomes 
test_data = "Mobile Price_test.csv"

# Load the trainig data, clean/prepare and obtain training and target vectors, 
def load_prepare():
    df = pd.read_csv(train_data)
    #X = df.drop('price_range', axis=1, inplace=False)
    X = df[['battery_power', 'clock_speed', 'fc', 'int_memory', 'mobile_wt', 'n_cores', 'pc', 'ram', 'talk_time']]
    y = df['price_range']
    return X, y

# Split it into train/validate sets
# Build a pipeline to transform the training vector and fit an appropriate machine learning model
# Validate your model accuracy using the validation set
def build_pipeline_1(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('std_scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=10))
        ])
    pipeline.fit(X_train,y_train)
    y_predict = pipeline.predict(X_test)
    training_accuracy = accuracy_score(y_test, y_predict).round(4)
    confusionmatrix = confusion_matrix(y_test, y_predict)
    return training_accuracy, confusionmatrix, pipeline
    

# Split it into train/validate sets
# Build a pipeline to transform the training vector and fit an appropriate machine learning model
# Validate your model accuracy using the validation set
def build_pipeline_2(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('std_scaler', StandardScaler()),
        ('dt',DecisionTreeClassifier(max_depth=6))
        ])
    pipeline.fit(X_train,y_train)
    y_predict = pipeline.predict(X_test)
    training_accuracy = accuracy_score(y_test, y_predict).round(4)
    confusionmatrix = confusion_matrix(y_test, y_predict)
    return training_accuracy, confusionmatrix, pipeline
    
  
# This your final and improved model pipeline
# Split it into train/validate sets
# Build a pipeline to transform the training vector and fit an appropriate machine learning model
# Validate your model accuracy using the validation set
# Save your final pipeline to a file "pipeline.pkl"   
def build_pipeline_final(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('std_scaler', StandardScaler()),
        ('lr',LogisticRegression(penalty='none', max_iter=1000))
        ])
    pipeline.fit(X_train,y_train)
    y_predict = pipeline.predict(X_test)
    training_accuracy = accuracy_score(y_test, y_predict).round(4)
    confusionmatrix = confusion_matrix(y_test, y_predict)
    pickle.dump(pipeline, open('final_pipeline.pkl','wb'))
    return training_accuracy, confusionmatrix, pipeline
   

# Load final pipeline (pipe.pkl) and test dataset (test_data)
# Apply the pipeline to the test data and predict outcomes
def apply_pipeline():
    pipeline = pickle.load(open('final_pipeline.pkl','rb'))
    df = pd.read_csv(test_data)
    df = df.drop('id',axis=1)
    df = df[['battery_power', 'clock_speed', 'fc', 'int_memory', 'mobile_wt', 'n_cores', 'pc', 'ram', 'talk_time']]
    predicted = pipeline.predict(df)
    df['price_range_predict'] = predicted
    return df
