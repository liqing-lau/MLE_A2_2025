"""
Model Monitoring DAG
Tracks model performance and data stability over time
"""

import sys
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import os

# Add scripts directory to Python path
sys.path.insert(0, "/opt/airflow")
sys.path.insert(0, "/opt/airflow/scripts")

PREDICTION_PATH = "/opt/airflow/scripts/datamart/gold/predictions/"
LABEL_PATH = "/opt/airflow/scripts/datamart/gold/label_store"
FEATURE_PATH = "/opt/airflow/scripts/datamart/gold/feature_store"
MONITORING_PATH = "/opt/airflow/scripts/datamart/gold/monitoring/"

os.makedirs(MONITORING_PATH, exist_ok=True)

default_args = {
    'owner': 'liqing_lau',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_monitoring_pipeline',
    default_args=default_args,
    description='Monitor model performance and data drift',
    schedule_interval='0 2 1 * *',  # Run on 1st of month at 2 AM
    start_date=datetime(2023, 1, 1),
    catchup=True,
    max_active_runs=1
)


def calculate_performance_metrics(**context):
    """Calculate model performance metrics when labels are available"""
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
    
    execution_date = context['ds']
    yyyy, mm = execution_date.split('-')[0], execution_date.split('-')[1]
    
    # Look back 6 months for labelled data (labels come with 30 day lag)
    monitor_date = datetime.strptime(execution_date, '%Y-%m-%d')
    lag_months = 2  # Assuming labels available after 2 months
    
    results = []
    
    for i in range(6):  # Last 6 months
        check_date = monitor_date - relativedelta(months=i+lag_months)
        y, m = check_date.strftime("%Y"), check_date.strftime("%m")
        
        pred_file = f"{PREDICTION_PATH}/predictions_{y}_{m}_01.parquet"
        label_file = f"{LABEL_PATH}/gold_label_{y}_{m}_01.parquet"
        
        if not os.path.exists(pred_file) or not os.path.exists(label_file):
            continue
        
        try:
            df_pred = pd.read_parquet(pred_file)
            df_label = pd.read_parquet(label_file)
            
            # Merge predictions and labels
            df = pd.merge(df_pred, df_label, on=['Customer_ID', 'loan_id'], how='inner')
            
            if len(df) == 0:
                continue
            
            y_true = df['label'].values
            y_pred = df['prediction'].values
            y_prob = df['probability'].values
            
            # Calculate metrics
            auc = roc_auc_score(y_true, y_prob)
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Prediction rate
            pred_rate = y_pred.mean()
            actual_rate = y_true.mean()
            
            results.append({
                'snapshot_date': check_date,
                'auc': auc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn,
                'prediction_rate': pred_rate,
                'actual_default_rate': actual_rate,
                'total_predictions': len(df),
                'monitoring_date': datetime.now()
            })
        except Exception as e:
            print(f"Error processing {check_date}: {e}")
            continue
    
    if results:
        df_metrics = pd.DataFrame(results)
        output_file = f"{MONITORING_PATH}/performance_metrics_{yyyy}_{mm}.parquet"
        df_metrics.to_parquet(output_file, index=False)
        print(f"✅ Performance metrics saved: {output_file}")
        print(df_metrics[['snapshot_date', 'auc', 'f1', 'prediction_rate']].to_string())
    else:
        print("⚠️ No labelled data available for performance monitoring (this is normal for early months)")


def calculate_feature_drift(**context):
    """Calculate feature distribution drift (PSI)"""
    
    execution_date = context['ds']
    yyyy, mm = execution_date.split('-')[0], execution_date.split('-')[1]
    
    current_date = datetime.strptime(execution_date, '%Y-%m-%d')
    baseline_date = current_date - relativedelta(months=6)
    
    # Load baseline and current features
    base_y, base_m = baseline_date.strftime("%Y"), baseline_date.strftime("%m")
    baseline_file = f"{FEATURE_PATH}/gold_features_{base_y}_{base_m}_01.parquet"
    current_file = f"{FEATURE_PATH}/gold_features_{yyyy}_{mm}_01.parquet"
    
    if not os.path.exists(baseline_file) or not os.path.exists(current_file):
        print(f"⚠️ Baseline or current features not found (baseline: {baseline_file}, current: {current_file})")
        print("This is normal for early months without 6 months of historical data")
        return
    
    try:
        df_baseline = pd.read_parquet(baseline_file)
        df_current = pd.read_parquet(current_file)
        
        # Drop ID columns
        id_cols = ["Customer_ID", "loan_id", "feature_snapshot_date"]
        baseline_features = df_baseline.drop(columns=id_cols)
        current_features = df_current.drop(columns=id_cols)
        
        # Calculate PSI for each feature
        def calculate_psi(baseline, current, bins=10):
            """Population Stability Index"""
            try:
                baseline_counts, bin_edges = np.histogram(baseline, bins=bins)
                current_counts, _ = np.histogram(current, bins=bin_edges)
                
                baseline_pct = baseline_counts / len(baseline) + 1e-10
                current_pct = current_counts / len(current) + 1e-10
                
                psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
                return psi
            except Exception as e:
                return np.nan
        
        psi_results = []
        for col in baseline_features.columns:
            psi = calculate_psi(baseline_features[col].dropna(), current_features[col].dropna())
            
            if np.isnan(psi):
                continue
            
            # Drift severity
            if psi < 0.1:
                severity = 'low'
            elif psi < 0.25:
                severity = 'medium'
            else:
                severity = 'high'
            
            psi_results.append({
                'feature': col,
                'psi': psi,
                'drift_severity': severity,
                'baseline_mean': baseline_features[col].mean(),
                'current_mean': current_features[col].mean(),
                'baseline_std': baseline_features[col].std(),
                'current_std': current_features[col].std(),
            })
        
        if psi_results:
            df_drift = pd.DataFrame(psi_results)
            df_drift['snapshot_date'] = current_date
            df_drift['baseline_date'] = baseline_date
            df_drift['monitoring_date'] = datetime.now()
            
            output_file = f"{MONITORING_PATH}/feature_drift_{yyyy}_{mm}.parquet"
            df_drift.to_parquet(output_file, index=False)
            
            print(f"✅ Feature drift saved: {output_file}")
            print(f"High drift features: {(df_drift['drift_severity'] == 'high').sum()}")
            high_drift = df_drift[df_drift['drift_severity'] == 'high']
            if len(high_drift) > 0:
                print(high_drift[['feature', 'psi']].to_string())
        else:
            print("⚠️ No valid PSI calculations (insufficient data)")
    
    except Exception as e:
        print(f"⚠️ Error calculating feature drift: {e}")


def calculate_prediction_stability(**context):
    """Monitor prediction distribution over time"""
    
    execution_date = context['ds']
    yyyy, mm = execution_date.split('-')[0], execution_date.split('-')[1]
    
    monitor_date = datetime.strptime(execution_date, '%Y-%m-%d')
    
    stability_results = []
    
    for i in range(6):  # Last 6 months
        check_date = monitor_date - relativedelta(months=i)
        y, m = check_date.strftime("%Y"), check_date.strftime("%m")
        
        pred_file = f"{PREDICTION_PATH}/predictions_{y}_{m}_01.parquet"
        
        if not os.path.exists(pred_file):
            continue
        
        try:
            df_pred = pd.read_parquet(pred_file)
            
            stability_results.append({
                'snapshot_date': check_date,
                'mean_probability': df_pred['probability'].mean(),
                'std_probability': df_pred['probability'].std(),
                'prediction_rate': df_pred['prediction'].mean(),
                'p10': df_pred['probability'].quantile(0.1),
                'p50': df_pred['probability'].quantile(0.5),
                'p90': df_pred['probability'].quantile(0.9),
                'total_predictions': len(df_pred),
                'monitoring_date': datetime.now()
            })
        except Exception as e:
            print(f"Error processing predictions for {check_date}: {e}")
            continue
    
    if stability_results:
        df_stability = pd.DataFrame(stability_results)
        output_file = f"{MONITORING_PATH}/prediction_stability_{yyyy}_{mm}.parquet"
        df_stability.to_parquet(output_file, index=False)
        print(f"✅ Prediction stability saved: {output_file}")
        print(df_stability[['snapshot_date', 'mean_probability', 'prediction_rate']].to_string())
    else:
        print("⚠️ No prediction data available for stability monitoring (this is normal for early months)")


def check_monitoring_data(**context):
    """Check if monitoring data is available"""
    import glob
    
    # Check for recent monitoring files
    perf_files = glob.glob(f"{MONITORING_PATH}/performance_metrics_*.parquet")
    drift_files = glob.glob(f"{MONITORING_PATH}/feature_drift_*.parquet")
    stab_files = glob.glob(f"{MONITORING_PATH}/prediction_stability_*.parquet")
    
    has_data = bool(perf_files or drift_files or stab_files)
    
    print(f"Found {len(perf_files)} performance files")
    print(f"Found {len(drift_files)} drift files")
    print(f"Found {len(stab_files)} stability files")
    
    if not has_data:
        print("⚠️ No monitoring data found - this is normal for early months without sufficient historical data")
        print("Skipping visualization and alerts for this run")
    
    # Push status to XCom so downstream tasks can check
    context['task_instance'].xcom_push(key='has_monitoring_data', value=has_data)
    
    return has_data


def generate_all_dashboards(**context):
    """Run the visualization script"""
    import sys
    import glob
    
    # Check if we have data first
    has_data = context['task_instance'].xcom_pull(task_ids='check_monitoring_data', key='has_monitoring_data')
    
    if not has_data:
        print("⚠️ Skipping dashboard generation - no monitoring data available")
        return
    
    sys.path.insert(0, "/opt/airflow/scripts")
    
    try:
        # Import and run visualization functions
        from scripts.utils.visualization import (
            create_performance_dashboard,
            create_drift_dashboard,
            create_stability_dashboard,
            generate_monitoring_report
        )
        
        print("Generating performance dashboard...")
        create_performance_dashboard()
        
        print("Generating drift dashboard...")
        create_drift_dashboard()
        
        print("Generating stability dashboard...")
        create_stability_dashboard()
        
        print("Generating monitoring report...")
        report = generate_monitoring_report()
        
        # Push report to XCom for potential email notification
        context['task_instance'].xcom_push(key='monitoring_report', value=report)
        
    except Exception as e:
        print(f"⚠️ Error generating dashboards: {e}")
        print("This may be due to insufficient data - continuing anyway")


def check_alerts(**context):
    """Check for alerts and send notifications if needed"""
    import pandas as pd
    import glob
    
    # Check if we have data first
    has_data = context['task_instance'].xcom_pull(task_ids='check_monitoring_data', key='has_monitoring_data')
    
    if not has_data:
        print("⚠️ Skipping alert checks - no monitoring data available")
        return
    
    alerts = []
    
    try:
        # Check performance degradation
        perf_files = sorted(glob.glob(f"{MONITORING_PATH}/performance_metrics_*.parquet"))
        if perf_files:
            dfs = [pd.read_parquet(f) for f in perf_files]
            df_perf = pd.concat(dfs, ignore_index=True).sort_values('snapshot_date')
            
            if len(df_perf) > 0:
                latest = df_perf.iloc[-1]
                
                if latest['auc'] < 0.70:
                    alerts.append(f"🚨 CRITICAL: AUC dropped to {latest['auc']:.4f} (threshold: 0.70)")
                elif latest['auc'] < 0.75:
                    alerts.append(f"⚠️ WARNING: AUC is {latest['auc']:.4f} (target: 0.75)")
                
                # Check for degradation
                if len(df_perf) > 1:
                    prev = df_perf.iloc[-2]
                    auc_change = latest['auc'] - prev['auc']
                    if auc_change < -0.05:
                        alerts.append(f"⚠️ Performance degradation: AUC dropped by {abs(auc_change):.4f}")
        
        # Check feature drift
        drift_files = sorted(glob.glob(f"{MONITORING_PATH}/feature_drift_*.parquet"))
        if drift_files:
            df_drift = pd.read_parquet(drift_files[-1])
            high_drift_count = (df_drift['drift_severity'] == 'high').sum()
            total_features = len(df_drift)
            
            if total_features > 0:
                high_drift_pct = high_drift_count / total_features
                
                if high_drift_pct > 0.30:
                    alerts.append(f"🚨 CRITICAL: {high_drift_count}/{total_features} features have high drift ({high_drift_pct:.1%})")
                elif high_drift_pct > 0.15:
                    alerts.append(f"⚠️ WARNING: {high_drift_count}/{total_features} features have high drift ({high_drift_pct:.1%})")
        
        # Check prediction stability
        stab_files = glob.glob(f"{MONITORING_PATH}/prediction_stability_*.parquet")
        if stab_files:
            dfs = [pd.read_parquet(f) for f in stab_files]
            df_stab = pd.concat(dfs, ignore_index=True).sort_values('snapshot_date')
            
            if len(df_stab) > 1:
                latest = df_stab.iloc[-1]
                prev = df_stab.iloc[-2]
                rate_change = abs(latest['prediction_rate'] - prev['prediction_rate'])
                
                if rate_change > 0.10:
                    alerts.append(f"🚨 CRITICAL: Prediction rate changed by {rate_change:.2%}")
                elif rate_change > 0.05:
                    alerts.append(f"⚠️ WARNING: Prediction rate changed by {rate_change:.2%}")
        
    except Exception as e:
        print(f"⚠️ Error checking alerts: {e}")
        return
    
    # Print and store alerts
    if alerts:
        print("\n" + "=" * 80)
        print("ALERTS DETECTED:")
        for alert in alerts:
            print(alert)
        print("=" * 80 + "\n")
        
        context['task_instance'].xcom_push(key='alerts', value=alerts)
        
        # In production, send email/Slack notification here
        # send_notification(alerts)
    else:
        print("✅ No alerts detected. Model performing within acceptable ranges.")


# Task definitions
performance_task = PythonOperator(
    task_id='calculate_performance_metrics',
    python_callable=calculate_performance_metrics,
    dag=dag,
)

drift_task = PythonOperator(
    task_id='calculate_feature_drift',
    python_callable=calculate_feature_drift,
    dag=dag,
)

stability_task = PythonOperator(
    task_id='calculate_prediction_stability',
    python_callable=calculate_prediction_stability,
    dag=dag,
)

check_data = PythonOperator(
    task_id='check_monitoring_data',
    python_callable=check_monitoring_data,
    dag=dag,
)

generate_viz = PythonOperator(
    task_id='generate_dashboards',
    python_callable=generate_all_dashboards,
    dag=dag,
)

check_alerts_task = PythonOperator(
    task_id='check_alerts',
    python_callable=check_alerts,
    dag=dag,
)

archive_old = BashOperator(
    task_id='archive_old_dashboards',
    bash_command="""
    DASHBOARD_PATH="/opt/airflow/scripts/dashboards/"
    ARCHIVE_PATH="/opt/airflow/scripts/dashboards/archive/"
    mkdir -p ${ARCHIVE_PATH}
    
    # Archive dashboards older than 30 days
    find ${DASHBOARD_PATH} -name "*.png" -mtime +30 -exec mv {} ${ARCHIVE_PATH} \; 2>/dev/null || true
    
    echo "Old dashboards archived (if any)"
    """,
    dag=dag,
)

# All monitoring tasks run independently, then check data, then visualize if data exists
[performance_task, drift_task, stability_task] >> check_data >> generate_viz >> check_alerts_task >> archive_old