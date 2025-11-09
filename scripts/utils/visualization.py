"""
Model Monitoring Visualization
Creates dashboards for model performance and stability tracking
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import glob

MONITORING_PATH = "/opt/airflow/scripts/datamart/gold/monitoring/"
DASHBOARD_PATH = "/opt/airflow/scripts/dashboards/"

os.makedirs(DASHBOARD_PATH, exist_ok=True)


def create_performance_dashboard():
    """Visualize model performance over time"""
    
    # Load all performance metrics
    files = glob.glob(f"{MONITORING_PATH}/performance_metrics_*.parquet")
    if not files:
        print("No performance metrics found")
        return
    
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('snapshot_date')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Monitoring Dashboard', fontsize=16, fontweight='bold')
    
    # AUC over time
    axes[0, 0].plot(df['snapshot_date'], df['auc'], marker='o', linewidth=2)
    axes[0, 0].axhline(y=0.7, color='r', linestyle='--', label='Threshold (0.7)')
    axes[0, 0].set_title('AUC Over Time')
    axes[0, 0].set_ylabel('AUC')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # F1 Score over time
    axes[0, 1].plot(df['snapshot_date'], df['f1'], marker='o', linewidth=2, color='green')
    axes[0, 1].set_title('F1 Score Over Time')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Precision vs Recall
    axes[1, 0].plot(df['snapshot_date'], df['precision'], marker='o', label='Precision', linewidth=2)
    axes[1, 0].plot(df['snapshot_date'], df['recall'], marker='s', label='Recall', linewidth=2)
    axes[1, 0].set_title('Precision vs Recall Over Time')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Prediction Rate vs Actual Default Rate
    axes[1, 1].plot(df['snapshot_date'], df['prediction_rate'], marker='o', label='Prediction Rate', linewidth=2)
    axes[1, 1].plot(df['snapshot_date'], df['actual_default_rate'], marker='s', label='Actual Default Rate', linewidth=2)
    axes[1, 1].set_title('Predicted vs Actual Default Rates')
    axes[1, 1].set_ylabel('Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{DASHBOARD_PATH}/performance_dashboard.png", dpi=300, bbox_inches='tight')
    print(f"Performance dashboard saved: {DASHBOARD_PATH}/performance_dashboard.png")


def create_drift_dashboard():
    """Visualize feature drift over time"""
    
    # Load latest drift metrics
    files = sorted(glob.glob(f"{MONITORING_PATH}/feature_drift_*.parquet"))
    if not files:
        print("No drift metrics found")
        return
    
    df = pd.read_parquet(files[-1])  # Latest
    
    # Sort by PSI descending
    df = df.sort_values('psi', ascending=False)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle(f'Feature Drift Analysis - {df["snapshot_date"].iloc[0]}', fontsize=16, fontweight='bold')
    
    # Top 20 features by PSI
    top_features = df.head(20)
    colors = top_features['drift_severity'].map({'low': 'green', 'medium': 'orange', 'high': 'red'})
    
    axes[0].barh(range(len(top_features)), top_features['psi'], color=colors)
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['feature'])
    axes[0].axvline(x=0.1, color='orange', linestyle='--', label='Medium Drift (0.1)')
    axes[0].axvline(x=0.25, color='red', linestyle='--', label='High Drift (0.25)')
    axes[0].set_xlabel('PSI')
    axes[0].set_title('Top 20 Features by Drift (PSI)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Drift severity distribution
    severity_counts = df['drift_severity'].value_counts()
    axes[1].bar(severity_counts.index, severity_counts.values, color=['green', 'orange', 'red'])
    axes[1].set_title('Feature Drift Severity Distribution')
    axes[1].set_ylabel('Number of Features')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{DASHBOARD_PATH}/drift_dashboard.png", dpi=300, bbox_inches='tight')
    print(f"Drift dashboard saved: {DASHBOARD_PATH}/drift_dashboard.png")


def create_stability_dashboard():
    """Visualize prediction stability over time"""
    
    # Load all stability metrics
    files = glob.glob(f"{MONITORING_PATH}/prediction_stability_*.parquet")
    if not files:
        print("No stability metrics found")
        return
    
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('snapshot_date')
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('Prediction Stability Dashboard', fontsize=16, fontweight='bold')
    
    # Mean probability over time with confidence bands
    axes[0].plot(df['snapshot_date'], df['mean_probability'], marker='o', linewidth=2, label='Mean')
    axes[0].fill_between(df['snapshot_date'], 
                         df['mean_probability'] - df['std_probability'],
                         df['mean_probability'] + df['std_probability'],
                         alpha=0.2, label='±1 Std Dev')
    axes[0].plot(df['snapshot_date'], df['p10'], linestyle='--', alpha=0.5, label='P10')
    axes[0].plot(df['snapshot_date'], df['p90'], linestyle='--', alpha=0.5, label='P90')
    axes[0].set_title('Prediction Probability Distribution Over Time')
    axes[0].set_ylabel('Probability')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Prediction rate over time
    axes[1].plot(df['snapshot_date'], df['prediction_rate'], marker='o', linewidth=2, color='orange')
    axes[1].set_title('Prediction Rate Over Time')
    axes[1].set_ylabel('Prediction Rate')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{DASHBOARD_PATH}/stability_dashboard.png", dpi=300, bbox_inches='tight')
    print(f"Stability dashboard saved: {DASHBOARD_PATH}/stability_dashboard.png")


def generate_monitoring_report():
    """Generate comprehensive monitoring report"""
    
    report = []
    report.append("=" * 80)
    report.append("MODEL MONITORING REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    
    # Performance metrics
    perf_files = glob.glob(f"{MONITORING_PATH}/performance_metrics_*.parquet")
    if perf_files:
        dfs = [pd.read_parquet(f) for f in perf_files]
        df_perf = pd.concat(dfs, ignore_index=True).sort_values('snapshot_date')
        latest_perf = df_perf.iloc[-1]
        
        report.append("\n1. LATEST PERFORMANCE METRICS")
        report.append(f"   Date: {latest_perf['snapshot_date']}")
        report.append(f"   AUC: {latest_perf['auc']:.4f}")
        report.append(f"   F1 Score: {latest_perf['f1']:.4f}")
        report.append(f"   Precision: {latest_perf['precision']:.4f}")
        report.append(f"   Recall: {latest_perf['recall']:.4f}")
        
        # Performance degradation check
        if len(df_perf) > 1:
            auc_change = latest_perf['auc'] - df_perf.iloc[-2]['auc']
            report.append(f"\n   Performance Change:")
            report.append(f"   AUC Change: {auc_change:+.4f}")
            if auc_change < -0.05:
                report.append("   ⚠️ WARNING: Significant AUC degradation detected!")
    
    # Feature drift
    drift_files = sorted(glob.glob(f"{MONITORING_PATH}/feature_drift_*.parquet"))
    if drift_files:
        df_drift = pd.read_parquet(drift_files[-1])
        high_drift = df_drift[df_drift['drift_severity'] == 'high']
        
        report.append("\n2. FEATURE DRIFT ANALYSIS")
        report.append(f"   Total Features: {len(df_drift)}")
        report.append(f"   High Drift: {len(high_drift)}")
        report.append(f"   Medium Drift: {(df_drift['drift_severity'] == 'medium').sum()}")
        report.append(f"   Low Drift: {(df_drift['drift_severity'] == 'low').sum()}")
        
        if len(high_drift) > 0:
            report.append("\n   ⚠️ Features with High Drift:")
            for _, row in high_drift.head(5).iterrows():
                report.append(f"   - {row['feature']}: PSI = {row['psi']:.4f}")
    
    # Prediction stability
    stab_files = glob.glob(f"{MONITORING_PATH}/prediction_stability_*.parquet")
    if stab_files:
        dfs = [pd.read_parquet(f) for f in stab_files]
        df_stab = pd.concat(dfs, ignore_index=True).sort_values('snapshot_date')
        latest_stab = df_stab.iloc[-1]
        
        report.append("\n3. PREDICTION STABILITY")
        report.append(f"   Mean Probability: {latest_stab['mean_probability']:.4f}")
        report.append(f"   Std Probability: {latest_stab['std_probability']:.4f}")
        report.append(f"   Prediction Rate: {latest_stab['prediction_rate']:.2%}")
        
        # Stability check
        if len(df_stab) > 1:
            rate_change = latest_stab['prediction_rate'] - df_stab.iloc[-2]['prediction_rate']
            if abs(rate_change) > 0.05:
                report.append(f"   ⚠️ WARNING: Prediction rate changed by {rate_change:+.2%}")
    
    report.append("\n" + "=" * 80)
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save report
    with open(f"{DASHBOARD_PATH}/monitoring_report.txt", 'w') as f:
        f.write(report_text)
    
    return report_text


if __name__ == "__main__":
    print("Generating monitoring dashboards...")
    create_performance_dashboard()
    create_drift_dashboard()
    create_stability_dashboard()
    generate_monitoring_report()
    print("\nAll dashboards generated successfully!")