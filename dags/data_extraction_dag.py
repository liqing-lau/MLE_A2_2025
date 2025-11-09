"""
Data Extraction DAG
Orchestrates bronze -> silver -> gold data processing pipeline
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import pyspark

# Add scripts directory to Python path
sys.path.insert(0, "/opt/airflow")
sys.path.insert(0, "/opt/airflow/scripts")

import scripts.utils.gold_processing as gp
import scripts.utils.bronze_processing as bp
import scripts.utils.silver_processing as sp

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
    'data_extraction_pipeline',
    default_args=default_args,
    description='Process data through bronze, silver, and gold layers',
    schedule_interval='0 0 1 * *',  # Run on 1st of every month at Midnight
    start_date=datetime(2023, 1, 1),
    catchup=True,  # Process historical dates
    max_active_runs=1
)

# Paths configuration
DATA_PATH = "/opt/airflow/scripts/data/"
BRONZE_DIR = "/opt/airflow/scripts/datamart/bronze/"
SILVER_DIR = "/opt/airflow/scripts/datamart/silver/"
GOLD_DIR = "/opt/airflow/scripts/datamart/gold/"

# File paths - All absolute
CLICKSTREAM_FILE = os.path.join(DATA_PATH, "feature_clickstream.csv")
ATTRIBUTES_FILE = os.path.join(DATA_PATH, "features_attributes.csv")
FINANCIALS_FILE = os.path.join(DATA_PATH, "features_financials.csv")
LOAN_DAILY_FILE = os.path.join(DATA_PATH, "lms_loan_daily.csv")

# Spark initialization helper
def get_spark():
    return pyspark.sql.SparkSession.builder \
        .appName("airflow_data_extraction") \
        .master("local[*]") \
        .getOrCreate()


# =============================================================================
# STATIC TABLES (Attributes & Financials) - Run once
# =============================================================================

def process_bronze_attributes(**context):
    spark = get_spark()
    try:
        bp.process_bronze_table(
            ATTRIBUTES_FILE, 
            BRONZE_DIR + "attribute/", 
            spark
        )
    finally:
        spark.stop()


def process_bronze_financials(**context):
    spark = get_spark()
    try:
        bp.process_bronze_table(
            FINANCIALS_FILE, 
            BRONZE_DIR + "financial/", 
            spark
        )
    finally:
        spark.stop()


def process_silver_attributes(**context):
    spark = get_spark()
    try:
        sp.process_silver_attribute_table(
            BRONZE_DIR + "attribute/",
            SILVER_DIR + "attribute/",
            spark
        )
    finally:
        spark.stop()


def process_silver_financials(**context):
    spark = get_spark()
    try:
        sp.process_silver_financial_table(
            BRONZE_DIR + "financial/",
            SILVER_DIR + "financial/",
            spark
        )
    finally:
        spark.stop()


def process_gold_attributes(**context):
    spark = get_spark()
    try:
        gp.process_gold_attribute_table(
            SILVER_DIR + "attribute/",
            GOLD_DIR + "attribute/",
            spark
        )
    finally:
        spark.stop()


def process_gold_financials(**context):
    spark = get_spark()
    try:
        gp.process_gold_finanical_table(
            SILVER_DIR + "financial/",
            GOLD_DIR + "financial/",
            spark
        )
    finally:
        spark.stop()


# =============================================================================
# DATE-BASED TABLES (Clickstream & LMS) - Run per execution date
# =============================================================================

def process_bronze_clickstream(**context):
    snapshot_date = context['ds']  # YYYY-MM-DD format
    spark = get_spark()
    try:
        bp.process_bronze_table_with_date(
            CLICKSTREAM_FILE,
            snapshot_date,
            BRONZE_DIR + "clickstream/",
            spark
        )
    finally:
        spark.stop()


def process_bronze_lms(**context):
    snapshot_date = context['ds']
    spark = get_spark()
    try:
        bp.process_bronze_table_with_date(
            LOAN_DAILY_FILE,
            snapshot_date,
            BRONZE_DIR + "lms/",
            spark
        )
    finally:
        spark.stop()


def process_silver_clickstream(**context):
    snapshot_date = context['ds']
    spark = get_spark()
    try:
        sp.process_silver_clickstream_table(
            snapshot_date,
            BRONZE_DIR + "clickstream/",
            SILVER_DIR + "clickstream/",
            spark
        )
    finally:
        spark.stop()


def process_silver_lms(**context):
    snapshot_date = context['ds']
    spark = get_spark()
    try:
        sp.process_silver_lms_table(
            snapshot_date,
            BRONZE_DIR + "lms/",
            SILVER_DIR + "lms/",
            spark
        )
    finally:
        spark.stop()


def process_gold_clickstream(**context):
    snapshot_date = context['ds']
    spark = get_spark()
    try:
        gp.process_gold_clickstream_table(
            snapshot_date,
            SILVER_DIR,
            GOLD_DIR,
            spark
        )
    finally:
        spark.stop()


def process_gold_feature_label(**context):
    snapshot_date = context['ds']
    spark = get_spark()
    try:
        gp.process_gold_feature_label_table(
            snapshot_date,
            SILVER_DIR,
            GOLD_DIR,
            spark,
            dpd=30,
            mob=6
        )
    finally:
        spark.stop()


# =============================================================================
# TASK DEFINITIONS
# =============================================================================

# Static tables - Attributes
bronze_attr = PythonOperator(
    task_id='bronze_attributes',
    python_callable=process_bronze_attributes,
    dag=dag,
)

silver_attr = PythonOperator(
    task_id='silver_attributes',
    python_callable=process_silver_attributes,
    dag=dag,
)

gold_attr = PythonOperator(
    task_id='gold_attributes',
    python_callable=process_gold_attributes,
    dag=dag,
)

# Static tables - Financials
bronze_fin = PythonOperator(
    task_id='bronze_financials',
    python_callable=process_bronze_financials,
    dag=dag,
)

silver_fin = PythonOperator(
    task_id='silver_financials',
    python_callable=process_silver_financials,
    dag=dag,
)

gold_fin = PythonOperator(
    task_id='gold_financials',
    python_callable=process_gold_financials,
    dag=dag,
)

# Date-based tables - Clickstream
bronze_click = PythonOperator(
    task_id='bronze_clickstream',
    python_callable=process_bronze_clickstream,
    dag=dag,
)

silver_click = PythonOperator(
    task_id='silver_clickstream',
    python_callable=process_silver_clickstream,
    dag=dag,
)

gold_click = PythonOperator(
    task_id='gold_clickstream',
    python_callable=process_gold_clickstream,
    dag=dag,
)

# Date-based tables - LMS
bronze_lms = PythonOperator(
    task_id='bronze_lms',
    python_callable=process_bronze_lms,
    dag=dag,
)

silver_lms = PythonOperator(
    task_id='silver_lms',
    python_callable=process_silver_lms,
    dag=dag,
)

# Final gold table
gold_feature_label = PythonOperator(
    task_id='gold_feature_label',
    python_callable=process_gold_feature_label,
    dag=dag,
)


# =============================================================================
# TASK DEPENDENCIES
# =============================================================================

# Attributes pipeline: bronze -> silver -> gold
bronze_attr >> silver_attr >> gold_attr

# Financials pipeline: bronze -> silver -> gold
bronze_fin >> silver_fin >> gold_fin

# Clickstream pipeline: bronze -> silver -> gold
bronze_click >> silver_click >> gold_click

# LMS pipeline: bronze -> silver
bronze_lms >> silver_lms

# Final feature_label needs:
# - gold attributes
# - gold financials  
# - gold clickstream
# - silver lms
[gold_attr, gold_fin, gold_click, silver_lms] >> gold_feature_label