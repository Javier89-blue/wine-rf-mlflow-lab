"""
Part B: Train a RandomForestClassifier on the FULL Wine dataset
using three different hyperparameter settings, and log every run to MLflow.
"""
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = "data/wine.csv"
DATASET_NAME = "wine"
DATASET_SETTING = "full_100"

# The three settings required by Part B of the assignment.
SETTINGS = [
    {"run_name": "RF_Setting_1_Full_Data", "n_estimators": 50, "max_depth": 3, "random_state": 42},
    {"run_name": "RF_Setting_2_Full_Data", "n_estimators": 100, "max_depth": 5, "random_state": 42},
    {"run_name": "RF_Setting_3_Full_Data", "n_estimators": 200, "max_depth": 7, "random_state": 42},
]


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]
    dataset_size = len(df)

    mlflow.set_experiment("wine-random-forest-lab")

    for setting in SETTINGS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=setting["random_state"],
            stratify=y,
        )

        with mlflow.start_run(run_name=setting["run_name"]):
            # Dataset metadata required by the assignment
            mlflow.log_param("dataset_name", DATASET_NAME)
            mlflow.log_param("dataset_size", dataset_size)
            mlflow.log_param("dataset_setting", DATASET_SETTING)

            # Model hyperparameters
            mlflow.log_param("n_estimators", setting["n_estimators"])
            mlflow.log_param("max_depth", setting["max_depth"])
            mlflow.log_param("random_state", setting["random_state"])

            model = RandomForestClassifier(
                n_estimators=setting["n_estimators"],
                max_depth=setting["max_depth"],
                random_state=setting["random_state"],
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)

            mlflow.sklearn.log_model(model, name="model-rf")

            print(f"[{setting['run_name']}] accuracy = {accuracy:.4f}")


if __name__ == "__main__":
    main()
