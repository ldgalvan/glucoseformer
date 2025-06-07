import argparse
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
import numpy as np

def process_and_save_cgm_data(input_file_path):
    """
    Full processing pipeline from raw data to saved sequences
    """
    spark = SparkSession.builder \
        .appName("CGM Data Processing") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()

    try:
        # 1. Load and filter data
        df = spark.read.option("header", "true") \
                      .option("delimiter", "|") \
                      .option("inferSchema", "true") \
                      .csv(input_file_path)

        # Filter valid timestamps
        timestamp_regex = r'^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} [AP]M$'
        df_filtered = df.filter(F.col("DeviceDtTm").rlike(timestamp_regex))

        print("\n=== Filter Results ===")
        print(f"Total filtered rows: {df_filtered.count()}")
        df_filtered.select("DeviceDtTm").show(5, truncate=False)

        # 2. Preprocess data
        df_processed = df_filtered.orderBy("PtID", "DeviceDtTm") \
            .fillna(0, subset=["BasalDelivPrev", "MealBolus", "BolusDelivPrev"]) \
            .withColumn("timestamp", F.to_timestamp("DeviceDtTm", "M/d/yyyy h:mm:ss a")) \
            .withColumn("date", F.to_date("timestamp"))

        # 3. Create sequences
        input_length = 120
        output_length = 12
        total_length = input_length + output_length

        daily_window = Window.partitionBy("PtID", "date").orderBy("timestamp")
        df_seq = df_processed.withColumn("daily_row_num", F.row_number().over(daily_window)) \
            .withColumn("max_sequences", F.floor((F.col("daily_row_num") - 1) / total_length)) \
            .filter(F.col("max_sequences") >= 0)

        df_sequences = df_seq.groupBy("PtID", "date", "max_sequences") \
            .agg(
                F.collect_list("CGMVal").alias("cgm_sequence"),
                F.collect_list("BasalDelivPrev").alias("basal_sequence"),
                F.collect_list("MealBolus").alias("meal_bolus_sequence"),
                F.collect_list("BolusDelivPrev").alias("bolus_deliv_prev_sequence"),
                F.collect_list("timestamp").alias("timestamp_sequence"),
                F.count("*").alias("sequence_length")
            ) \
            .filter(
                (F.col("sequence_length") == total_length) &
                (F.size("cgm_sequence") == total_length) &
                (F.size("basal_sequence") == total_length) &
                (F.size("meal_bolus_sequence") == total_length) &
                (F.size("bolus_deliv_prev_sequence") == total_length)
            )

        # 4. Convert to numpy arrays
        def create_sequences(rows):
            X, y = [], []
            for row in rows:
                try:
                    # Validate single-day sequence
                    if len({ts.date() for ts in row.timestamp_sequence}) != 1:
                        continue
                        
                    # Convert to arrays
                    full_cgm = np.array(row.cgm_sequence)
                    basal = np.array(row.basal_sequence)
                    meal_bolus = np.array(row.meal_bolus_sequence)
                    bolus_deliv_prev = np.array(row.bolus_deliv_prev_sequence)
                    
                    # Stack features (4 channels)
                    X.append(np.column_stack((
                        full_cgm[:input_length],
                        basal[:input_length],
                        meal_bolus[:input_length],
                        bolus_deliv_prev[:input_length]
                    )))
                    
                    y.append(full_cgm[input_length:input_length+output_length])
                except Exception as e:
                    continue
            return zip(X, y)

        sequences_rdd = df_sequences.rdd.mapPartitions(create_sequences)
        X, y = zip(*sequences_rdd.collect())

        X = np.array(X) if len(X) > 0 else np.empty((0, input_length, 4))
        y = np.array(y) if len(y) > 0 else np.empty((0, output_length))

        # 5. Temporal split
        total_samples = len(X)
        train_end = int(total_samples * 0.8)
        val_end = train_end + int(total_samples * 0.1)

        splits = {
            'X_train': X[:train_end],
            'y_train': y[:train_end],
            'X_val': X[train_end:val_end],
            'y_val': y[train_end:val_end],
            'X_test': X[val_end:],
            'y_test': y[val_end:]
        }

        # 6. Save results
        np.savez("cgm_sequences10.npz", **splits)

        print(f"\n✅ Final splits:")
        print(f"Train: {splits['X_train'].shape} | {splits['y_train'].shape}")
        print(f"Val:   {splits['X_val'].shape} | {splits['y_val'].shape}")
        print(f"Test:  {splits['X_test'].shape} | {splits['y_test'].shape}")
        print("\nSaved to cgm_sequences.npz in current directory")

    except Exception as e:
        print(f"\n❌ Error processing data: {str(e)}")
        raise

    finally:
        spark.stop()
        print("\nSpark session stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and split CGM data for ML training.")
    parser.add_argument("input_file", type=str, help="Path to IOBP2DeviceiLet.txt file")
    args = parser.parse_args()
    process_and_save_cgm_data(args.input_file)

