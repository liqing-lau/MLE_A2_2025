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

def process_silver_clickstream_table(snapshot_date_str, bronze_dir, silver_dir, spark): 
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    filename = "bronze_feature_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_dir + filename
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {"fe_1": IntegerType(),
                       'fe_2': IntegerType(),
                       'fe_3': IntegerType(),
                       'fe_4': IntegerType(),
                       'fe_5': IntegerType(),
                       'fe_6': IntegerType(),
                       'fe_7': IntegerType(),
                       'fe_8': IntegerType(),
                       'fe_9': IntegerType(),
                       'fe_10': IntegerType(),
                       'fe_11': IntegerType(),
                       'fe_12': IntegerType(),
                       'fe_13': IntegerType(),
                       'fe_14': IntegerType(),
                       'fe_15': IntegerType(),
                       'fe_16': IntegerType(),
                       'fe_17': IntegerType(),
                       'fe_18': IntegerType(),
                       'fe_19': IntegerType(),
                       'fe_20': IntegerType(),
                       "Customer_ID": StringType(),
                       "snapshot_date": DateType()
                    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    if not os.path.exists(silver_dir):
        os.makedirs(silver_dir)

    # save silver table - IRL connect to database to write
    filename = "silver_feature_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_dir + filename
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df

def process_silver_attribute_table(bronze_dir, silver_dir, spark): 
    filepath = bronze_dir + "bronze_features_attributes.csv"
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: strip and replace alphanumeric for name, age and ssn
    df = df.withColumn("Name", F.trim(F.col("Name")))
    df = df.withColumn("Name", F.regexp_replace("Name", "[^A-Za-z ]", ""))
    df = df.withColumn("SSN", F.regexp_replace(F.col("SSN"), r"\s+", ""))
    df = df.withColumn("Age", F.regexp_replace("Age", "[^0-9]", ""))

    df = df.withColumn("Occupation", 
                       F.when(F.col("Occupation") == "_______", "Unknown")
                       .otherwise(F.col("Occupation")))

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {"Customer_ID": StringType(),
                       "Name": StringType(),
                       'Age': IntegerType(),
                       'SSN': StringType(),
                       "Occupation": StringType(),
                       "snapshot_date": DateType()
                      }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: duplicated ssn
    w = Window.partitionBy("SSN")
    df = df.withColumn("duplicate_ssn", (F.count("*").over(w) > 1).cast(BooleanType()))
    
    # augment data: duplicated name and ssn
    w = Window.partitionBy("Name", "SSN")
    df = df.withColumn("duplicate_name_ssn", (F.count("*").over(w) > 1).cast(BooleanType()))

    # augment data: potential age typo (bleow 18 and above 100)
    df = df.withColumn("age_typo", (F.col("Age") < 0) | (F.col("Age") > 100).cast(BooleanType()))
    
    if not os.path.exists(silver_dir):
        os.makedirs(silver_dir)

    # save silver table - IRL connect to database to write
    filepath = silver_dir + "silver_feature_attributes.parquet"
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df

def process_silver_financial_table(bronze_dir, silver_dir, spark): 
    filepath = bronze_dir + "bronze_features_financials.csv"
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    df = df.withColumn("Changed_Credit_Limit", 
                       F.when(F.col("Changed_Credit_Limit") == "_", 0)
                       .otherwise(F.col("Changed_Credit_Limit")))

    df = df.withColumn("Type_of_Loan", 
                       F.when(F.col("Type_of_Loan").isNull(), "not specified")
                       .otherwise(F.col("Type_of_Loan")))
    df = df.withColumn("Type_of_Loan",
                       F.lower(F.trim(F.regexp_replace("Type_of_Loan", "and ", ""))))
    
    df = df.withColumn("Credit_Mix", 
                       F.when(F.col("Credit_Mix") == "_", None)
                       .otherwise(F.col("Credit_Mix")))
    
    df = df.withColumn("Payment_Behaviour", 
                       F.when(F.col("Payment_Behaviour") == "!@9#%8", "Unknown")
                       .otherwise(F.col("Payment_Behaviour")))

    # clean data: enforce schema / data type
    cleaning_rules = {"Annual_Income": "[^0-9.]",
                      "Num_of_Loan": r"[^0-9.\-]",
                      "Num_of_Delayed_Payment": r"[^0-9.\-]",
                      "Changed_Credit_Limit": r"[^0-9.\-]",
                      "Outstanding_Debt": "[^0-9.]",
                      "Amount_invested_monthly": "[^0-9.]",
                      "Monthly_Balance": "[^0-9.]",
                    }
    for column, value in cleaning_rules.items(): 
        df = df.withColumn(column, F.regexp_replace(column, value, ""))
    
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {'Customer_ID': StringType(),
                        'Annual_Income': DoubleType(),
                        'Monthly_Inhand_Salary': DoubleType(),
                        'Num_Bank_Accounts': DoubleType(),
                        'Num_Credit_Card': DoubleType(),
                        'Interest_Rate': DoubleType(),
                        'Num_of_Loan': DoubleType(),
                        'Type_of_Loan': StringType(),
                        'Delay_from_due_date': DoubleType(),
                        'Num_of_Delayed_Payment': DoubleType(),
                        'Changed_Credit_Limit': DoubleType(),
                        'Num_Credit_Inquiries': DoubleType(),
                        'Credit_Mix': StringType(),
                        'Outstanding_Debt': DoubleType(),
                        'Credit_Utilization_Ratio': DoubleType(),
                        'Credit_History_Age': StringType(),
                        'Payment_of_Min_Amount': StringType(),
                        'Total_EMI_per_month': DoubleType(),
                        'Amount_invested_monthly': DoubleType(),
                        'Payment_Behaviour': StringType(),
                        'Monthly_Balance': DoubleType(),
                        'snapshot_date': DateType()
                        }

    for column, new_type in column_type_map.items():          
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: negative values
    df = df.withColumn("Negative_Changed_Credit_Limit", 
                       (F.col("Changed_Credit_Limit") < 0).cast(BooleanType()))
    
    df = df.withColumn("Negative_Delay_from_due_date", 
                       (F.col("Delay_from_due_date") < 0).cast(BooleanType()))

    df = df.withColumn("No_bank_accounts", 
                       (F.col("Num_Bank_Accounts") == 0).cast(BooleanType()))

    # augment data: potential typo
    df = df.withColumn("Num_of_Delayed_Payment_typo", (F.col("Num_of_Delayed_Payment") < 0) | 
                       (F.col("Num_of_Delayed_Payment") > 9).cast(BooleanType()))
    
    df = df.withColumn("Num_of_Loan_typo", (F.col("Num_of_Loan") < 0) | 
                       (F.col("Num_of_Loan") > 9).cast(BooleanType()))

    df = df.withColumn("Num_Credit_Card_typo", (F.col("Num_Credit_Card") < 0) | 
                       (F.col("Num_Credit_Card") > 11).cast(BooleanType()))

    df = df.withColumn("Num_Bank_Accounts_typo", (F.col("Num_Bank_Accounts") < 0) | 
                       (F.col("Num_Bank_Accounts") > 11).cast(BooleanType()))

    df = df.withColumn("Interest_Rate_typo", (F.col("Interest_Rate") < 0) | 
                       (F.col("Interest_Rate") > 11).cast(BooleanType()))

    df = df.withColumn("Num_of_Delayed_Payment_typo", (F.col("Num_of_Delayed_Payment") < 0) | 
                       (F.col("Num_of_Delayed_Payment") > 28).cast(BooleanType()))

    df = df.withColumn("Num_Credit_Inquiries_typo", (F.col("Num_Credit_Inquiries") < 0) | 
                       (F.col("Num_Credit_Inquiries") > 17).cast(BooleanType()))

    # augment data: annual income outlier
    df = df.withColumn("Annual_Income_outlier", (F.col("Annual_Income") > 100_000).cast(BooleanType()))

    # augment data: get Credit_History_Months in months
    df = df.withColumn("Credit_History_Years",
                       F.coalesce(
                       F.regexp_extract("Credit_History_Age", r"(\d+)\s+Years?", 1).cast(DoubleType()),
                       F.lit(0)))
                       
    df = df.withColumn("Credit_History_Months",
                       F.coalesce(
                       F.regexp_extract("Credit_History_Age", r"(\d+)\s+Months?", 1).cast(DoubleType()),
                       F.lit(0)))
                        
    df = df.withColumn("Credit_History_Total_Months",
                       ((F.col("Credit_History_Years") * 12) + F.col("Credit_History_Months")).cast(DoubleType())
                      )
    
    # augment data: get outstanding debt to monthly income ratio
    df = df.withColumn("Debt_to_Annual_Income_Ratio", 
                       (F.col("Outstanding_Debt") / F.col("Annual_Income")).cast(DoubleType()))

    # augment data: get outstanding debt to monthly income ratio
    df = df.withColumn("Debt_to_Monthly_Income_Ratio", 
                       (F.col("Outstanding_Debt") / F.col("Monthly_Inhand_Salary")).cast(DoubleType()))

    df = df.withColumn("Payment_of_Min_Amount_clean",
                       F.when(F.lower(F.col("Payment_of_Min_Amount")) == "yes", True)
                       .when(F.lower(F.col("Payment_of_Min_Amount")) == "no", False)
                       .otherwise(False))

    # split payment behaviour
    df = df.withColumn("Spend_Level",
                       F.when(F.col("Payment_Behaviour") == "Unknown", "Unknown")
                       .otherwise(F.split(F.col("Payment_Behaviour"), "_")[0]))
    
    df = df.withColumn("Payment_Value",
                       F.when(F.col("Payment_Behaviour") == "Unknown", "Unknown")
                       .otherwise(F.split(F.col("Payment_Behaviour"), "_")[2]))

    # create array for loan types and remove not specified if there are other values in the array
    df = df.withColumn("Loan_Array", F.split(F.col("Type_of_Loan"), ","))
    df = df.withColumn("Loan_Array", F.expr("transform(Loan_Array, x -> trim(x))"))
    df = df.withColumn("Loan_Array",F.expr("""
                                            CASE 
                                                WHEN size(Loan_Array) > 1 
                                                THEN filter(Loan_Array, x -> x != 'not specified') 
                                                ELSE Loan_Array 
                                            END
                                        """))
    
    if not os.path.exists(silver_dir):
        os.makedirs(silver_dir)
    
    # save silver table - IRL connect to database to write
    filepath = silver_dir + "silver_feature_financials.parquet"
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df

def process_silver_lms_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_lms_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {"loan_id": StringType(),
                       "Customer_ID": StringType(),
                       "loan_start_date": DateType(),
                       "tenure": IntegerType(),
                       "installment_num": IntegerType(),
                       "loan_amt": FloatType(),
                       "due_amt": FloatType(),
                       "paid_amt": FloatType(),
                       "overdue_amt": FloatType(),
                       "balance": FloatType(),
                       "snapshot_date": DateType(),
                       }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_lms_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df