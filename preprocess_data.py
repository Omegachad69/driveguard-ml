import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# File paths
raw_data_path = os.path.join("data", "raw", "train_motion_data.csv")
processed_data_path = os.path.join("data", "processed", "preprocessed_data.csv")

# Load raw data
raw_data = pd.read_csv(raw_data_path)


# Feature Engineering
def add_features(df):
    # Magnitude of acceleration and gyroscope
    df["Acc_Magnitude"] = np.sqrt(df["AccX"] ** 2 + df["AccY"] ** 2 + df["AccZ"] ** 2)
    df["Gyro_Magnitude"] = np.sqrt(
        df["GyroX"] ** 2 + df["GyroY"] ** 2 + df["GyroZ"] ** 2
    )

    # Rolling statistics for accelerometer
    window_size = 5
    df["AccX_Mean"] = df["AccX"].rolling(window=window_size).mean()
    df["AccX_Std"] = df["AccX"].rolling(window=window_size).std()
    df["AccY_Mean"] = df["AccY"].rolling(window=window_size).mean()
    df["AccY_Std"] = df["AccY"].rolling(window=window_size).std()
    df["AccZ_Mean"] = df["AccZ"].rolling(window=window_size).mean()
    df["AccZ_Std"] = df["AccZ"].rolling(window=window_size).std()

    # Drop rows with NaN values (due to rolling window)
    df = df.dropna()
    return df


# Preprocessing steps
def preprocess_data(df, target_column, timestamp_column=None):
    # Drop the timestamp column if it exists
    if timestamp_column and timestamp_column in df.columns:
        df = df.drop(columns=[timestamp_column])

    # Encode the target column if it's categorical
    if df[target_column].dtype == "object":
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])

    # Normalize the numerical sensor columns
    sensor_columns = [col for col in df.columns if col != target_column]
    scaler = StandardScaler()
    df[sensor_columns] = scaler.fit_transform(df[sensor_columns])

    return df


# Add features and preprocess data
raw_data = add_features(raw_data)
processed_data = preprocess_data(
    raw_data, target_column="Class", timestamp_column="Timestamp"
)

# Save the processed data to a CSV file
os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
processed_data.to_csv(processed_data_path, index=False)

print(f"Processed data saved to {processed_data_path}")
