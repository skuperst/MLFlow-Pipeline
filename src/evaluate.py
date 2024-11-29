import mlflow
from mlflow.models import infer_signature

import os
from urllib.parse import urlparse

import yaml

import pandas as pd

import pickle

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV

with open('.env', 'r') as file:
    for line in file:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, value = line.split('=', 1)
        os.environ[key] = value

USER_NAME = os.getenv('MY_USER_NAME')
PASSWORD = os.getenv('MY_PASSWORD')

my_tracking_uri = "https://dagshub.com/skuperst/MLFlow_Pipeline.mlflow"

os.environ['MLFLOW_TRACKING_URI'] = my_tracking_uri
os.environ['MLFLOW_TRACKING_USERNAME'] = USER_NAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = PASSWORD

# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["evaluate"]

def evaluate(data_path, model_path):

    # Load the preprocessed data
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri(my_tracking_uri)

    # Load the model
    model=pickle.load(open(model_path,'rb'))

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    # Log metrics to MLFLOW
    mlflow.log_metric("accuracy", accuracy)
    print("Model accuracy:{accuracy}")

if __name__=="__main__":
    evaluate(data_path = params["data"], model_path = params["model"])