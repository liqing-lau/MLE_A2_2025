import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import argparse
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler, StandardScaler

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, BooleanType, DoubleType

import utils.gold_processing
import utils.bronze_processing
import utils.silver_processing

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

path = "data/"
clickstream = path + "feature_clickstream.csv"
attributes = path + "features_attributes.csv"
financials = path + "features_financials.csv"
loan_daily = path + "lms_loan_daily.csv"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)

utils.bronze_processing.process_bronze_table(attributes, "datamart/bronze/attribute/", spark)
utils.bronze_processing.process_bronze_table(financials, "datamart/bronze/financial/", spark)
utils.silver_processing.process_silver_attribute_table("datamart/bronze/attribute/", 
                                                        "datamart/silver/attribute/", 
                                                        spark)
utils.silver_processing.process_silver_financial_table("datamart/bronze/financial/", 
                                                        "datamart/silver/financial/", 
                                                        spark)
utils.gold_processing.process_gold_attribute_table("datamart/silver/attribute/", "datamart/gold/attribute/", spark)
utils.gold_processing.process_gold_finanical_table("datamart/silver/financial/", "datamart/gold/financial/", spark)

for snapshotdate in dates_str_lst:
    utils.bronze_processing.process_bronze_table_with_date(clickstream, snapshotdate, "datamart/bronze/clickstream/", spark)
    utils.bronze_processing.process_bronze_table_with_date(loan_daily, snapshotdate, "datamart/bronze/lms/", spark)

    utils.silver_processing.process_silver_clickstream_table(snapshotdate, 
                                                            "datamart/bronze/clickstream/", 
                                                            "datamart/silver/clickstream/", 
                                                            spark)
    utils.silver_processing.process_silver_lms_table(snapshotdate, 
                                                    "datamart/bronze/lms/", 
                                                    "datamart/silver/lms/", 
                                                    spark)
    utils.gold_processing.process_gold_clickstream_table(snapshotdate, "datamart/silver/", "datamart/gold/", spark)
    utils.gold_processing.process_gold_feature_label_table(snapshotdate, "datamart/silver/", "datamart/gold/", spark,  dpd = 30, mob = 6)