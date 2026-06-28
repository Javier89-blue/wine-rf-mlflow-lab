"""
Part D: Retrain the BEST parameter setting (selected in Part C) on the
80% REDUCED Wine dataset, and log the run to MLflow.
"""
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = "data/wine_reduced_80.csv"
DATASET_NAME = "wine"
DATASET_SETTING = "reduced_80"


def load_best_params(path="best_params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    best_params = load_best_params()

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]
    dataset_size = len(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=best_params["random_state"],
        stratify=y,
    )

    mlflow.set_experiment("wine-random-forest-lab")

    with mlflow.start_run(run_name="Best_RF_Reduced_80_Data"):
        mlflow.log_param("dataset_name", DATASET_NAME)
        mlflow.log_param("dataset_size", dataset_size)
        mlflow.log_param("dataset_setting", DATASET_SETTING)

        mlflow.log_param("n_estimators", best_params["n_estimators"])
        mlflow.log_param("max_depth", best_params["max_depth"])
        mlflow.log_param("random_state", best_params["random_state"])

        model = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            random_state=best_params["random_state"],
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(model, name="model-rf")

        print(f"[Best_RF_Reduced_80_Data] accuracy = {accuracy:.4f}")


if __name__ == "__main__":
    main()
