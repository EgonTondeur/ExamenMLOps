import argparse
import json
import os

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

import mlflow
import mlflow.sklearn


def main():
    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument("--train_ready", type=str, required=True)
    parser.add_argument("--test_ready", type=str, required=True)

    # Hyperparameters
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--target_col", type=str, default="house_affiliation")

    # Outputs
    parser.add_argument("--model_output", type=str, required=True)
    parser.add_argument("--metrics_output", type=str, required=True)

    args = parser.parse_args()

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    X_train_path = os.path.join(args.train_ready, "X_train.csv")
    y_train_path = os.path.join(args.train_ready, "y_train.csv")
    X_test_path  = os.path.join(args.test_ready, "X_test.csv")
    y_test_path  = os.path.join(args.test_ready, "y_test.csv")

    for p in [X_train_path, y_train_path, X_test_path, y_test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)[args.target_col].astype(str)

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)[args.target_col].astype(str)

    # --------------------------------------------------
    # Train model
    # --------------------------------------------------
    clf = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )

    clf.fit(X_train, y_train)

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    # --------------------------------------------------
    # MLflow logging (CRUCIAAL voor register_model)
    # --------------------------------------------------
    mlflow.start_run()

    # Log parameters
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
    mlflow.log_param("random_state", args.random_state)

    # Log metrics
    mlflow.log_metric("accuracy", acc)

    # Log full classification report
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{label}_{metric_name}", value)

    # Log model as MLflow model
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="mlflow_model"
    )

    mlflow.end_run()

    # --------------------------------------------------
    # Save metrics artifact (optioneel, maar netjes)
    # --------------------------------------------------
    os.makedirs(args.metrics_output, exist_ok=True)
    with open(os.path.join(args.metrics_output, "metrics.json"), "w") as f:
        json.dump(
            {
                "accuracy": acc,
                "classification_report": report
            },
            f,
            indent=2
        )

    print("✅ Training completed successfully")
    print("Accuracy:", acc)


if __name__ == "__main__":
    main()
