# modelling.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from helper import load_and_split
import mlflow

data_path = "transactions_preprocessing/metaverse_clean.csv"
X_train, X_test, y_train, y_test = load_and_split(data_path)

with mlflow.start_run():

    mlflow.log_param('n_estimators', 505)
    mlflow.log_param('max_depth', 37)

    model = RandomForestClassifier(
        n_estimators=505,
        max_depth=37
    )

    mlflow.autolog()

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    mlflow.log_metric('accuracy', accuracy)

