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

# Random Forest Grid search
def hyperparameter_tuning(X_train,y_train, param_grid):
    rf=RandomForestClassifier()
    grid_search=GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

## Load the parameters from params.yaml
curr_dir = os.path.dirname(os.path.abspath(__file__)) 
params=yaml.safe_load(open(os.path.join(curr_dir, '..', "params.yaml")))["train"]

def train(data_path,model_path, random_state, n_estimators, max_depth):

    # Load the preprocessed data
    data = pd.read_csv(data_path)
    output_column = 'Outcome'
    X = data.drop(columns=[output_column])
    y = data[output_column]

    # Start MLFlow tracking
    mlflow.set_tracking_uri(my_tracking_uri)

    # Start the MLFlow run
    with mlflow.start_run():

        # Split the preprocessed data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

        # Signature and Input exmple
        signature=infer_signature(X_train, y_train)
        input_example = X_train.head(1)

        ## Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # Perform hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Predict and evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy:{accuracy}")

        # Log additional metrics
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_param("best_n_estimatios", grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
        mlflow.log_param("best_sample_split", grid_search.best_params_['min_samples_split'])
        mlflow.log_param("best_samples_leaf", grid_search.best_params_['min_samples_leaf'])

        ## Log Confusion Matrix and Classification Report

        cm = confusion_matrix(y_test,y_pred)
        cr = classification_report(y_test,y_pred)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")

        if urlparse(mlflow.get_tracking_uri()).scheme != 'file':
            mlflow.sklearn.log_model(sk_model=best_model, artifact_path="model", registered_model_name="Best Model", 
                                     input_example=input_example, signature=signature)
        else:
            mlflow.sklearn.log_model(sk_model=best_model, artifact_path="model", 
                                     input_example=input_example, signature=signature)

        # Create the directory to save the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save the model
        pickle.dump(best_model,open(model_path,'wb'))

        print(f"Model saved to {model_path}")

if __name__=="__main__":
    train(data_path = params['data'],model_path = params['model'], random_state = params['random_state'], 
          n_estimators = params['n_estimators'], max_depth = params['max_depth'])


