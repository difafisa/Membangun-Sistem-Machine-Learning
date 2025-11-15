from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import mlflow
from helper import load_and_split
from sklearn.metrics import classification_report, precision_score, f1_score

mlflow.set_tracking_uri("https://dagshub.com/difafisa/Membangun-Sistem-Machine-Learning.mlflow")
mlflow.set_experiment("Metaverse_RF_Tuning")


data_path = "preprocessing/transactions_preprocessing/metaverse_clean.csv"
X_train, X_test, y_train, y_test = load_and_split(data_path)

with mlflow.start_run():
    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid = RandomizedSearchCV(rf, param_grid, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # === METRICS ===
    acc = best_model.score(X_test, y_test)
    macro_precision = precision_score(y_test, y_pred, average='macro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("macro_precision", macro_precision)
    mlflow.log_metric("macro_f1", macro_f1)

    # === CLASSIFICATION REPORT (ARTIFACT) ===
    report = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")

    # === SAVE MODEL ===
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_rf_model"
    )
    
        


