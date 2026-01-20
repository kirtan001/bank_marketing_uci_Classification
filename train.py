import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from imblearn.under_sampling import RandomUnderSampler

# --- FIX 1: Robust Tracking URI ---
# Ensure absolute path is used to avoid relative path confusion on Linux runners
mlflow_dir = os.path.abspath("mlruns")
os.makedirs(mlflow_dir, exist_ok=True)
mlflow.set_tracking_uri(f"file://{mlflow_dir}")

df = pd.read_csv("cleaned_bank_data.csv")

X = df.drop(columns=["y"])
y = df["y"]

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rus = RandomUnderSampler(random_state=42)
X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)
X_test_bal, y_test_bal = rus.fit_resample(X_test, y_test)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

mlflow.set_experiment("Bank_Marketing_Tree_Models")

results = {}
models = {}
run_ids = {}  # Store run_ids to register the best model later

# --- Model 1: Decision Tree ---
with mlflow.start_run(run_name="Decision_Tree") as run:
    dt = DecisionTreeClassifier(max_depth=6, min_samples_split=50, random_state=42)
    dt.fit(X_train_bal, y_train_bal)
    metrics = evaluate_model(dt, X_test_bal, y_test_bal)
    
    mlflow.log_params({"max_depth": 6, "min_samples_split": 50})
    mlflow.log_metrics(metrics)
    
    # --- FIX 2: Explicit artifact_path to fix deprecation warning ---
    mlflow.sklearn.log_model(dt, artifact_path="model")
    
    results["Decision Tree"] = metrics
    models["Decision Tree"] = dt
    run_ids["Decision Tree"] = run.info.run_id

# --- Model 2: Random Forest ---
with mlflow.start_run(run_name="Random_Forest") as run:
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=50, n_jobs=-1, random_state=42)
    rf.fit(X_train_bal, y_train_bal)
    metrics = evaluate_model(rf, X_test_bal, y_test_bal)
    
    mlflow.log_params({"n_estimators": 200, "max_depth": 10, "min_samples_split": 50})
    mlflow.log_metrics(metrics)
    
    mlflow.sklearn.log_model(rf, artifact_path="model")
    
    results["Random Forest"] = metrics
    models["Random Forest"] = rf
    run_ids["Random Forest"] = run.info.run_id

# --- Model 3: XGBoost ---
with mlflow.start_run(run_name="XGBoost") as run:
    scale_pos_weight = y_train_bal.value_counts()[0] / y_train_bal.value_counts()[1]
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric="logloss")
    
    xgb.fit(X_train_bal, y_train_bal)
    metrics = evaluate_model(xgb, X_test_bal, y_test_bal)
    
    mlflow.log_params({"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05})
    mlflow.log_metrics(metrics)
    
    mlflow.xgboost.log_model(xgb, artifact_path="model")
    
    results["XGBoost"] = metrics
    models["XGBoost"] = xgb
    run_ids["XGBoost"] = run.info.run_id

# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv("model_comparison.csv")

# --- FIX 3: Register Best Model Correctly ---
# Find best model name
best_model_name = results_df["roc_auc"].idxmax()
best_model = models[best_model_name]
best_run_id = run_ids[best_model_name]  # Retrieve the correct run_id

print(f"Best Model Selected: {best_model_name} (Run ID: {best_run_id})")

os.makedirs("artifacts", exist_ok=True)
joblib.dump(best_model, "artifacts/best_model.pkl")
joblib.dump(X_train.columns.tolist(), "artifacts/feature_columns.pkl")

# Register the model using the explicitly stored Run ID
mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name="BankMarketingBestModel"
)