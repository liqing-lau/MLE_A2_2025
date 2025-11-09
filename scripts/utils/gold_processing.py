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

def process_gold_attribute_table(silver_dir, gold_dir, spark):
    filepath = silver_dir + "silver_feature_attributes.parquet"

    df = spark.read.parquet(filepath)

    print('loaded from:', filepath, 'row count:', df.count())

    # impute average age 
    mean_age = mean_age = df.filter(~F.col("age_typo")).agg(F.avg("Age")).collect()[0][0]

    df = df.withColumn("Age",
                       F.when(F.col("age_typo"), F.lit(mean_age)) 
                       .otherwise(F.col("Age"))
                       .cast(IntegerType())
                       )

    # label encode occupation
    indexer = StringIndexer(inputCol="Occupation", outputCol="Occupation_Label")
    df = indexer.fit(df).transform(df)

    # remove name, ssn and age_typo columns
    df = df.drop("Name", "SSN", 'age_typo', 'Occupation')

    # save gold table - IRL connect to database to write
    filepath = gold_dir + "gold_feature_attributes.parquet"
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df

def process_gold_finanical_table(silver_dir, gold_dir, spark):
    filepath = silver_dir + "silver_feature_financials.parquet"

    df = spark.read.parquet(filepath)

    print('loaded from:', filepath, 'row count:', df.count())

    # augment data: potential typo
    mean_loan = df.filter(~F.col("Num_of_Loan_typo")).agg(F.avg("Num_of_Loan")).collect()[0][0]

    df = df.withColumn("Num_of_Loan",
                       F.when(F.col("Num_of_Loan_typo"), F.lit(mean_loan)) 
                       .otherwise(F.col("Num_of_Loan"))
                       .cast(IntegerType())
                       )
    
    mean_cc = df.filter(~F.col("Num_Credit_Card_typo")).agg(F.avg("Num_Credit_Card")).collect()[0][0]

    df = df.withColumn("Num_Credit_Card",
                       F.when(F.col("Num_Credit_Card_typo"), F.lit(mean_cc)) 
                       .otherwise(F.col("Num_Credit_Card"))
                       .cast(IntegerType())
                       )

    mean_bank_acc = df.filter(~F.col("Num_Bank_Accounts_typo")).agg(F.avg("Num_Bank_Accounts")).collect()[0][0]

    df = df.withColumn("Num_Bank_Accounts",
                       F.when(F.col("Num_Bank_Accounts_typo"), F.lit(mean_bank_acc)) 
                       .otherwise(F.col("Num_Bank_Accounts"))
                       .cast(IntegerType())
                       )

    mean_interest_rate = df.filter(~F.col("Interest_Rate_typo")).agg(F.avg("Interest_Rate")).collect()[0][0]

    df = df.withColumn("Interest_Rate",
                       F.when(F.col("Interest_Rate_typo"), F.lit(mean_interest_rate)) 
                       .otherwise(F.col("Interest_Rate"))
                       .cast(IntegerType())
                       )

    mean_delayed_payment = df.filter(~F.col("Num_of_Delayed_Payment_typo")).agg(F.avg("Num_of_Delayed_Payment")).collect()[0][0]

    df = df.withColumn("Num_of_Delayed_Payment",
                       F.when(F.col("Num_of_Delayed_Payment_typo"), F.lit(mean_delayed_payment)) 
                       .otherwise(F.col("Num_of_Delayed_Payment"))
                       .cast(IntegerType())
                       )

    mean_credit_inquiries = df.filter(~F.col("Num_Credit_Inquiries_typo")).agg(F.avg("Num_Credit_Inquiries")).collect()[0][0]

    df = df.withColumn("Num_Credit_Inquiries",
                       F.when(F.col("Num_Credit_Inquiries_typo"), F.lit(mean_credit_inquiries)) 
                       .otherwise(F.col("Num_Credit_Inquiries"))
                       .cast(IntegerType())
                       )

    # high debt ratio
    df = df.withColumn("Monthly_Income_Level", 
                       F.when(F.col("Monthly_Inhand_Salary") <= 2500, "Low")
                       .when(F.col("Monthly_Inhand_Salary") > 10000, "High")
                       .otherwise("Medium"))
    indexer_spend = StringIndexer(inputCol="Monthly_Income_Level", outputCol="Monthly_Income_Level_index")
    df = indexer_spend.fit(df).transform(df)
    
    # Spend_Level
    indexer_spend = StringIndexer(inputCol="Spend_Level", outputCol="Spend_Level_index")
    df = indexer_spend.fit(df).transform(df)
    
    # Payment_Value
    indexer_payment = StringIndexer(inputCol="Payment_Value", outputCol="Payment_Value_index")
    df = indexer_payment.fit(df).transform(df)
    
    # Credit_Mix
    indexer_credit = StringIndexer(inputCol="Credit_Mix", outputCol="Credit_Mix_index", handleInvalid="keep" )
    df = indexer_credit.fit(df).transform(df)

    # Payment_Behaviour
    indexer_credit = StringIndexer(inputCol="Payment_Behaviour", outputCol="Payment_Behaviour_index", handleInvalid="keep" )
    df = indexer_credit.fit(df).transform(df)

    # Loan_Array
    df_exploded = df.withColumn("Loan_Type", F.explode("Loan_Array"))
    df_loans_ohe = (df_exploded.groupBy("Customer_ID").pivot("Loan_Type").agg(F.lit(1)).fillna(0))
    df = df.join(df_loans_ohe, on="Customer_ID", how="left")
    
    # high debt ratio
    df = df.withColumn("High_Debt_Monthly_Salary_Ratio", 
                       (F.col("Debt_to_Monthly_Income_Ratio") > 4).cast(BooleanType()))

    df = df.drop("Type_of_Loan", "Credit_Mix", 'Payment_of_Min_Amount', 'Credit_History_Age',
                'Payment_Behaviour', 'Spend_Level', "Num_of_Delayed_Payment_typo", "Num_of_Delayed_Payment"
                 'Payment_Value', 'Loan_Array', 'Monthly_Income_Level', 'Num_Credit_Card_typo',
                "Num_of_Loan_typo", "Interest_Rate_typo", 'Num_of_Delayed_Payment_typo', 'Num_Credit_Inquiries_typo',
                'Negative_Delay_from_due_date', 'Delay_from_due_date', "Payment_of_Min_Amount_clean",
                'Payment_Value', 'not specified')

    # save gold table - IRL connect to database to write
    filepath = gold_dir + "gold_feature_financials.parquet"
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df

def process_gold_clickstream_table(snapshot_date_str, silver_dir, gold_dir, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    clickstream_filename = "silver_feature_clickstream_"  + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_dir + "clickstream/" + clickstream_filename
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    attribute = gold_dir + "attribute/gold_feature_attributes.parquet"
    attribute_df = spark.read.parquet(attribute)
    
    financial = gold_dir + "financial/gold_feature_financials.parquet"
    financial_df = spark.read.parquet(financial)

    attribute_df = attribute_df.withColumnRenamed("snapshot_date", "attr_snapshot_date")
    financial_df = financial_df.withColumnRenamed("snapshot_date", "fin_snapshot_date")
    
    # combined financial and attributes if financial's snapshot date is >= attribute's
    attr_fin = financial_df.join(attribute_df, 
                                 "Customer_ID", 
                                 "inner").filter(F.col("attr_snapshot_date") <= F.col("fin_snapshot_date"))
    w = Window.partitionBy("Customer_ID", "fin_snapshot_date").orderBy(F.col("attr_snapshot_date").desc())
    attr_fin = attr_fin.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") == 1).drop("rn", "attr_snapshot_date")

    attr_fin_click = attr_fin.join(df, 
                                   "Customer_ID", 
                                   "inner").filter(F.col("fin_snapshot_date") <= F.col("snapshot_date"))
    
    # Keep snapshot_date as feature_snapshot_date for temporal validation in downstream processes
    attr_fin_click = attr_fin_click.withColumnRenamed("snapshot_date", "feature_snapshot_date").drop("fin_snapshot_date")

    filepath = gold_dir + "combined_feature/gold_combined_features_"+ snapshot_date_str.replace('-','_') + ".parquet"
    attr_fin_click.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath, 'row count:', attr_fin_click.count())
    
    return attr_fin_click

def process_gold_feature_label_table(snapshot_date_str, silver_dir, gold_dir, spark, dpd, mob): 
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_lms_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_dir + "lms/" + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    feature_filename = "gold_combined_features_"  + snapshot_date_str.replace('-','_') + '.parquet'
    feature_filepath = gold_dir + "combined_feature/" + feature_filename
    feature_df = spark.read.parquet(feature_filepath)

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))

    # select columns for label table
    label_df = df.select("loan_id", "Customer_ID", "label", "snapshot_date")

    # TEMPORAL VALIDATION: Ensure features are not from the future relative to loan data
    # Join with temporal check - feature_snapshot_date must be <= loan snapshot_date
    validated_join = label_df.join(feature_df, "Customer_ID", "inner").filter(
        F.col("feature_snapshot_date") <= F.col("snapshot_date")
    )
    
    print(f"Rows before temporal filter: {label_df.count()}")
    print(f"Rows after temporal filter: {validated_join.count()}")
    
    # 1. Label Store (Y) - loan_id, Customer_ID, label, label_def, snapshot_date
    label_store = validated_join.select("loan_id", "Customer_ID", "label", "snapshot_date")
    label_filepath = gold_dir + "label_store/gold_label_"+ snapshot_date_str.replace('-','_') + ".parquet"
    label_store.write.mode("overwrite").parquet(label_filepath)
    print(f'Label store saved to: {label_filepath}, row count: {label_store.count()}')
    
    # 2. Feature Store (X) - loan_id, Customer_ID, all features, feature_snapshot_date
    # Drop label columns to keep only features
    feature_store = validated_join.drop("label", "label_def", "snapshot_date")
    feature_filepath = gold_dir + "feature_store/gold_features_"+ snapshot_date_str.replace('-','_') + ".parquet"
    feature_store.write.mode("overwrite").parquet(feature_filepath)
    print(f'Feature store saved to: {feature_filepath}, row count: {feature_store.count()}')

    return {
        'label_store': label_store,
        'feature_store': feature_store
    }