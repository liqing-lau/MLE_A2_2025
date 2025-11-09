"""
Model Inference DAG
Makes predictions on new data and stores results in gold layer
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pickle
import os

FEATURE_PATH = "/opt/airflow/scripts/datamart/gold/feature_store"
MODEL_PATH = "/opt/airflow/scripts/model_store/"
PREDICTION_PATH = "/opt/airflow/scripts/datamart/gold/predictions/"

os.makedirs(PREDICTION_PATH, exist_ok=True)

default_args = {
    'owner': 'liqing_lau',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_inference_pipeline',
    default_args=default_args,
    description='Generate predictions on latest features',
    schedule_interval='0 1 1 * *',  # Run after extraction at 1 AM
    start_date=datetime(2023, 1, 1),
    catchup=True,
    max_active_runs=1
)


def make_predictions(**context):
    """Load latest features and generate predictions"""
    
    execution_date = context['ds']
    yyyy, mm = execution_date.split('-')[0], execution_date.split('-')[1]
    
    # Load latest features
    feature_file = f"{FEATURE_PATH}/gold_features_{yyyy}_{mm}_01.parquet"
    if not os.path.exists(feature_file):
        print(f"Feature file not found: {feature_file}")
        return
    
    df_features = pd.read_parquet(feature_file)
    
    # Load model and scaler
    with open(f"{MODEL_PATH}/model.pkl", 'rb') as f:
        model = pickle.load(f)
    with open(f"{MODEL_PATH}/scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    # Prepare features
    id_cols = ["Customer_ID", "loan_id", "feature_snapshot_date"]
    ids = df_features[id_cols].copy()
    
    x_arr = df_features.drop(columns=id_cols).values
    x_scaled = scaler.transform(x_arr)
    
    # Generate predictions
    predictions = model.predict(x_scaled)
    probabilities = model.predict_proba(x_scaled)[:, 1]
    
    # Create results dataframe
    results = ids.copy()
    results['prediction'] = predictions
    results['probability'] = probabilities
    results['model_version'] = context['run_id']
    results['prediction_date'] = datetime.now()
    
    # Save predictions
    output_file = f"{PREDICTION_PATH}/predictions_{yyyy}_{mm}_01.parquet"
    results.to_parquet(output_file, index=False)
    
    print(f"Predictions saved: {output_file}")
    print(f"Total predictions: {len(results)}")
    print(f"Predicted positives: {predictions.sum()} ({predictions.mean():.2%})")


# Task definition
predict = PythonOperator(
    task_id='make_predictions',
    python_callable=make_predictions,
    dag=dag,
)