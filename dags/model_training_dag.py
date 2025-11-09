"""
Model Training DAG
Trains and evaluates ML models on gold layer features
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

# Paths
FEATURE_PATH = "/opt/airflow/scripts/datamart/gold/feature_store"
LABEL_PATH = "/opt/airflow/scripts/datamart/gold/label_store"
MODEL_PATH = "/opt/airflow/scripts/model_store/"
TEMP_PATH = "/opt/airflow/scripts/temp/"

# Ensure directories exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(TEMP_PATH, exist_ok=True)

# DAG configuration
default_args = {
    'owner': 'liqing_lau',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='Train ML models on feature store data',
    schedule_interval="@yearly",
    start_date=datetime(2025, 11, 1),
    catchup=False, 
    max_active_runs=1
)


def prepare_dataset(**context):
    """Load last 18 months of data and split into train/test/oot"""
    
    # Find all available months
    files = sorted([f for f in os.listdir(FEATURE_PATH) if f.startswith("gold_features_")])
    months = []
    for f in files:
        y, m = f.split("_")[2], f.split("_")[3]
        months.append(datetime(int(y), int(m), 1))
    
    months = sorted(months)
    latest_month = months[-1]
    
    # Last 18 months including latest
    selected_months = [latest_month - relativedelta(months=i) for i in range(18)]
    selected_months = sorted(selected_months)
    
    oot_month = selected_months[-1]
    train_test_months = selected_months[:-1]
    
    # Load train/test data
    feature_list, label_list = [], []
    for m in train_test_months:
        yyyy, mm = m.strftime("%Y"), m.strftime("%m")
        df_features = pd.read_parquet(f"{FEATURE_PATH}/gold_features_{yyyy}_{mm}_01.parquet")
        df_labels = pd.read_parquet(f"{LABEL_PATH}/gold_label_{yyyy}_{mm}_01.parquet")
        feature_list.append(df_features)
        label_list.append(df_labels)
    
    x_df = pd.concat(feature_list, ignore_index=True)
    y_df = pd.concat(label_list, ignore_index=True)
    
    # Load OOT data
    yyyy, mm = oot_month.strftime("%Y"), oot_month.strftime("%m")
    x_oot_df = pd.read_parquet(f"{FEATURE_PATH}/gold_features_{yyyy}_{mm}_01.parquet")
    y_oot_df = pd.read_parquet(f"{LABEL_PATH}/gold_label_{yyyy}_{mm}_01.parquet")
    
    # Split train/test
    x_train, x_test, y_train, y_test = train_test_split(
        x_df, y_df, test_size=0.20, stratify=y_df['label'], random_state=42
    )
    
    # Save to temp files
    x_train.to_parquet(f"{TEMP_PATH}/x_train.parquet")
    x_test.to_parquet(f"{TEMP_PATH}/x_test.parquet")
    x_oot_df.to_parquet(f"{TEMP_PATH}/x_oot.parquet")
    y_train.to_parquet(f"{TEMP_PATH}/y_train.parquet")
    y_test.to_parquet(f"{TEMP_PATH}/y_test.parquet")
    y_oot_df.to_parquet(f"{TEMP_PATH}/y_oot.parquet")
    
    print(f"Train: {x_train.shape}, Test: {x_test.shape}, OOT: {x_oot_df.shape}")


def scale_features(**context):
    """Scale features using StandardScaler"""
    
    # Load data
    x_train = pd.read_parquet(f"{TEMP_PATH}/x_train.parquet")
    x_test = pd.read_parquet(f"{TEMP_PATH}/x_test.parquet")
    x_oot = pd.read_parquet(f"{TEMP_PATH}/x_oot.parquet")
    
    # Drop ID columns
    x_train_arr = x_train.drop(columns=["Customer_ID", "loan_id", "feature_snapshot_date"]).values
    x_test_arr = x_test.drop(columns=["Customer_ID", "loan_id", "feature_snapshot_date"]).values
    x_oot_arr = x_oot.drop(columns=["Customer_ID", "loan_id", "feature_snapshot_date"]).values
    
    # Scale
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_arr)
    x_test_scaled = scaler.transform(x_test_arr)
    x_oot_scaled = scaler.transform(x_oot_arr)
    
    # Save scaled data and scaler
    np.save(f"{TEMP_PATH}/x_train_scaled.npy", x_train_scaled)
    np.save(f"{TEMP_PATH}/x_test_scaled.npy", x_test_scaled)
    np.save(f"{TEMP_PATH}/x_oot_scaled.npy", x_oot_scaled)
    
    with open(f"{MODEL_PATH}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)


def train_logistic_regression(**context):
    """Grid search and train Logistic Regression"""
    
    # Load data
    x_train_scaled = np.load(f"{TEMP_PATH}/x_train_scaled.npy")
    y_train = pd.read_parquet(f"{TEMP_PATH}/y_train.parquet")["label"].values
    
    # Grid search
    log_reg = LogisticRegression(max_iter=500)
    log_reg_params = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"]
    }
    
    log_reg_grid = GridSearchCV(
        estimator=log_reg,
        param_grid=log_reg_params,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )
    
    log_reg_grid.fit(x_train_scaled, y_train)
    
    print(f"Best LR Params: {log_reg_grid.best_params_}")
    print(f"Best LR Score: {log_reg_grid.best_score_:.4f}")
    
    # Train final model
    best_lr = LogisticRegression(**log_reg_grid.best_params_)
    best_lr.fit(x_train_scaled, y_train)
    
    with open(f"{TEMP_PATH}/lr_model.pkl", 'wb') as f:
        pickle.dump(best_lr, f)
    
    # Push best score to XCom
    context['task_instance'].xcom_push(key='lr_score', value=log_reg_grid.best_score_)


def train_random_forest(**context):
    """Grid search and train Random Forest"""
    
    # Load data
    x_train_scaled = np.load(f"{TEMP_PATH}/x_train_scaled.npy")
    y_train = pd.read_parquet(f"{TEMP_PATH}/y_train.parquet")["label"].values
    
    # Grid search
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 50],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }
    
    rf_grid = GridSearchCV(
        estimator=rf,
        param_grid=rf_params,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )
    
    rf_grid.fit(x_train_scaled, y_train)
    
    print(f"Best RF Params: {rf_grid.best_params_}")
    print(f"Best RF Score: {rf_grid.best_score_:.4f}")
    
    # Train final model
    best_rf = RandomForestClassifier(**rf_grid.best_params_)
    best_rf.fit(x_train_scaled, y_train)
    
    with open(f"{TEMP_PATH}/rf_model.pkl", 'wb') as f:
        pickle.dump(best_rf, f)
    
    # Push best score to XCom
    context['task_instance'].xcom_push(key='rf_score', value=rf_grid.best_score_)


def evaluate_models(**context):
    """Evaluate both models and select the best"""
    
    # Load data
    x_train_scaled = np.load(f"{TEMP_PATH}/x_train_scaled.npy")
    x_test_scaled = np.load(f"{TEMP_PATH}/x_test_scaled.npy")
    x_oot_scaled = np.load(f"{TEMP_PATH}/x_oot_scaled.npy")
    
    y_train = pd.read_parquet(f"{TEMP_PATH}/y_train.parquet")["label"].values
    y_test = pd.read_parquet(f"{TEMP_PATH}/y_test.parquet")["label"].values
    y_oot = pd.read_parquet(f"{TEMP_PATH}/y_oot.parquet")["label"].values
    
    # Load models
    with open(f"{TEMP_PATH}/lr_model.pkl", 'rb') as f:
        lr_model = pickle.load(f)
    with open(f"{TEMP_PATH}/rf_model.pkl", 'rb') as f:
        rf_model = pickle.load(f)
    
    # Evaluate LR
    y_pred_train_lr = lr_model.predict(x_train_scaled)
    y_pred_test_lr = lr_model.predict(x_test_scaled)
    y_pred_oot_lr = lr_model.predict(x_oot_scaled)
    
    train_auc_lr = roc_auc_score(y_train, y_pred_train_lr)
    test_auc_lr = roc_auc_score(y_test, y_pred_test_lr)
    oot_auc_lr = roc_auc_score(y_oot, y_pred_oot_lr)
    
    train_f1_lr = f1_score(y_train, y_pred_train_lr)
    test_f1_lr = f1_score(y_test, y_pred_test_lr)
    oot_f1_lr = f1_score(y_oot, y_pred_oot_lr)
    
    # Evaluate RF
    y_pred_train_rf = rf_model.predict(x_train_scaled)
    y_pred_test_rf = rf_model.predict(x_test_scaled)
    y_pred_oot_rf = rf_model.predict(x_oot_scaled)
    
    train_auc_rf = roc_auc_score(y_train, y_pred_train_rf)
    test_auc_rf = roc_auc_score(y_test, y_pred_test_rf)
    oot_auc_rf = roc_auc_score(y_oot, y_pred_oot_rf)
    
    train_f1_rf = f1_score(y_train, y_pred_train_rf)
    test_f1_rf = f1_score(y_test, y_pred_test_rf)
    oot_f1_rf = f1_score(y_oot, y_pred_oot_rf)
    
    # Print results
    print(f"\n============LOGISTIC REGRESSION============")
    print(f"Train AUC: {train_auc_lr:.4f}, F1: {train_f1_lr:.4f}")
    print(f"Test AUC: {test_auc_lr:.4f}, F1: {test_f1_lr:.4f}")
    print(f"OOT AUC: {oot_auc_lr:.4f}, F1: {oot_f1_lr:.4f}")
    
    print(f"\n============RANDOM FOREST============")
    print(f"Train AUC: {train_auc_rf:.4f}, F1: {train_f1_rf:.4f}")
    print(f"Test AUC: {test_auc_rf:.4f}, F1: {test_f1_rf:.4f}")
    print(f"OOT AUC: {oot_auc_rf:.4f}, F1: {oot_f1_rf:.4f}")
    
    # Select best model based on test AUC
    if test_auc_rf > test_auc_lr:
        print(f"\nRandom Forest selected (Test AUC: {test_auc_rf:.4f})")
        with open(f"{MODEL_PATH}/model.pkl", 'wb') as f:
            pickle.dump(rf_model, f)
    else:
        print(f"\nLogistic Regression selected (Test AUC: {test_auc_lr:.4f})")
        with open(f"{MODEL_PATH}/model.pkl", 'wb') as f:
            pickle.dump(lr_model, f)

def cleanup_temp_files(**context):
    """Remove temporary files after training completes"""
    import shutil
    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)
        os.makedirs(TEMP_PATH)
    print("Temporary files cleaned up")


# Task definitions
prepare_data = PythonOperator(
    task_id='prepare_dataset',
    python_callable=prepare_dataset,
    dag=dag,
)

scale_data = PythonOperator(
    task_id='scale_features',
    python_callable=scale_features,
    dag=dag,
)

train_lr = PythonOperator(
    task_id='train_logistic_regression',
    python_callable=train_logistic_regression,
    dag=dag,
)

train_rf = PythonOperator(
    task_id='train_random_forest',
    python_callable=train_random_forest,
    dag=dag,
)

evaluate = PythonOperator(
    task_id='evaluate_models',
    python_callable=evaluate_models,
    dag=dag,
)

cleanup = PythonOperator(
    task_id='cleanup_temp_files',
    python_callable=cleanup_temp_files,
    dag=dag,
)

# Task dependencies
prepare_data >> scale_data >> [train_lr, train_rf] >> evaluate >> cleanup