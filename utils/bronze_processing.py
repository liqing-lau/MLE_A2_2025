import os
from datetime import datetime

from pyspark.sql.functions import col

def process_bronze_table_with_date(filepath, snapshot_date_str, bronze_directory, spark): 
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    filename = filepath.split("/")[-1][:-4]
    
    # load data
    df = spark.read.csv(filepath, 
                        header=True, 
                        inferSchema=True).filter(col('snapshot_date') == snapshot_date)

    # save bronze table to datamart
    partition_name = "bronze_" + filename + "_" + snapshot_date_str.replace('-','_') + '.csv'
    bronze_filepath = bronze_directory + partition_name

    # make directory if it doesnt exists
    if not os.path.exists(bronze_directory):
        os.makedirs(bronze_directory)
        
    df.toPandas().to_csv(bronze_filepath, index=False)
    print('saved to:', bronze_filepath)

    return df

def process_bronze_table(filepath, bronze_directory, spark): 
    filename = filepath.split("/")[-1][:-4]
    
    # load data
    df = spark.read.csv(filepath, 
                        header=True, 
                        inferSchema=True)

    # save bronze table to datamart
    partition_name = "bronze_" + filename + '.csv'
    bronze_filepath = bronze_directory + partition_name

    # make directory if it doesnt exists
    if not os.path.exists(bronze_directory):
        os.makedirs(bronze_directory)
        
    df.toPandas().to_csv(bronze_filepath, index=False)
    print('saved to:', bronze_filepath)

    return df