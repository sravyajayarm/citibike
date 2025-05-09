# Citi Bike Pipeline using Hopsworks Feature Store
import pandas as pd
import hopsworks
from hsfs.feature import Feature
import importlib
import mlflow
import mlflow.sklearn
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

# 🔐 Replace with your actual API key and project name
API_KEY = "dPfPCqiDr7gFNPWt.ZOzP1t4FjexKpiALxnCtm7JWH1xwIjIi5cZMWu3Fkfm44jlH7IGz2GXDRvY5kU2S"  # Hopsworks API key
PROJECT_NAME = "bike"  # Hopsworks project name
FEATURE_GROUP_NAME = "cbtpsc_cleaned_v1"  # Fresh feature group name
INPUT_CSV = "/content/202504-citibike-tripdata_1.csv"  # Upload this to Colab first

# DagsHub Authentication Setup
DAGSHUB_USERNAME = "sravyajaogulamba"  # Replace with your DagsHub username
DAGSHUB_REPO = "citibike-prediction"  # Replace with your DagsHub repository name
DAGSHUB_TOKEN = "fd7a0acc39a062d3d86631cb97a6080ded2fbc3a"  # Replace with your DagsHub token

# Tracking URI for DagsHub MLflow
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")

# Set the authentication token in environment variables
os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN

# Create or use existing experiment in DagsHub
mlflow.set_experiment("cbtpsc-experiments")  # This experiment will be created in DagsHub

print("✅ Connected to DagsHub MLflow Tracking")

# Load data
def load_data(file_path):
    print(f"📥 Loading data from {file_path} ...")
    df = pd.read_csv(
        file_path,
        dtype={
            'start_station_id': str,
            'end_station_id': str,
            'ride_id': str,
            'rideable_type': str,
            'start_station_name': str,
            'end_station_name': str,
            'member_casual': str
        },
        parse_dates=['started_at', 'ended_at'],
        low_memory=False,
    )
    df.columns = df.columns.str.strip()
    print("✅ Data loaded. Columns:", df.columns.tolist())
    return df

# Select top 3 start stations
def select_top_locations(df):
    top3 = df['start_station_name'].value_counts().nlargest(3).index.tolist()
    print(f"🏙️ Top 3 start stations: {top3}")
    return top3

# Preprocess data
def clean_and_preprocess_data(df, top_locations):
    print("🧼 Preprocessing data...")
    df = df[df['start_station_name'].isin(top_locations)].copy()
    df.dropna(inplace=True)
    df['trip_duration'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60
    df.reset_index(drop=True, inplace=True)
    print("✅ Preprocessing complete.")
    return df

# Save data to CSV
def save_data_to_csv(df, output_file="cleaned_citi_bike_data.csv"):
    if df.empty:
        print("⚠️ DataFrame is empty. Skipping save.")
        return
    df.to_csv(output_file, index=False)
    print(f"📁 Data saved to {output_file}.")

# Store data in Hopsworks
def store_data_in_hopsworks(df, api_key_value, project_name, feature_group_name):
    try:
        project = hopsworks.login(api_key_value=api_key_value, project=project_name)
        print(f"✅ Logged into Hopsworks project: {project.name}")
        fs = project.get_feature_store()

        features = [
            Feature("ride_id", "string"),
            Feature("rideable_type", "string"),
            Feature("started_at", "timestamp"),
            Feature("ended_at", "timestamp"),
            Feature("start_station_name", "string"),
            Feature("start_station_id", "string"),
            Feature("end_station_name", "string"),
            Feature("end_station_id", "string"),
            Feature("start_lat", "double"),
            Feature("start_lng", "double"),
            Feature("end_lat", "double"),
            Feature("end_lng", "double"),
            Feature("member_casual", "string"),
            Feature("trip_duration", "double"),
        ]

        # 🔁 Always delete FG if exists (to avoid corrupt schema)
        try:
            fg_existing = fs.get_feature_group(name=feature_group_name, version=1)
            if fg_existing:
                fg_existing.delete()
                print(f"🧹 Deleted existing feature group '{feature_group_name}' (version 1)")
        except:
            print(f"ℹ️ No existing feature group to delete.")

        # ✅ Create new FG
        fg = fs.create_feature_group(
            name=feature_group_name,
            version=1,
            description="Cleaned Citi Bike data with trip duration",
            primary_key=["ride_id"],
            event_time="started_at",
            features=features,
            online_enabled=True
        )
        print(f"📊 Feature group '{feature_group_name}' created.")

        # 🔄 Ensure types match
        for feat in features:
            col = feat.name
            if col in df.columns:
                if feat.type == 'string':
                    df[col] = df[col].astype(str)
                elif feat.type == 'timestamp':
                    df[col] = pd.to_datetime(df[col])
                elif feat.type == 'double':
                    df[col] = pd.to_numeric(df[col])

        # 🚀 Insert data
        fg.insert(df, write_options={"wait_for_job": True})
        print("✅ Data inserted into Feature Group successfully.")

    except Exception as e:
        print("❌ Error inserting data into Hopsworks:", e)

# STEP 2a: Create 28-day lag features
def create_lag_features(df, days=28):
    print(f"🧼 Creating {days}-day lag features...")
    
    df['pickup_date'] = df['started_at'].dt.date
    df = df.sort_values(by=['start_station_name', 'pickup_date'])

    # Create lag features (shift the trip_duration for past 'days' days)
    for i in range(1, days + 1):
        df[f'lag_{i}'] = df.groupby('start_station_name')['trip_duration'].shift(i)
    
    df.dropna(inplace=True)  # Drop rows with missing lag values
    print(f"✅ Created {days}-day lag features.")
    return df

# STEP 2b: Train LightGBM Model and Log in MLflow
def log_lgbm_model(df):
    print("🧑‍💻 Training LightGBM model...")
    
    # Ensure lag features are created
    df_with_lags = create_lag_features(df, days=28)
    
    # Select features (lags)
    feature_cols = [f'lag_{i}' for i in range(1, 29)]
    
    X = df_with_lags[feature_cols]
    y = df_with_lags['trip_duration']
    
    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train LightGBM model
    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)
    
    # Predict and calculate MAE
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model MAE: {mae}")
    
    # Log the model and its metrics in MLflow
    with mlflow.start_run():
        mlflow.log_metric("mae_lgbm", mae)
        mlflow.log_param("model", "LightGBM")
        mlflow.sklearn.log_model(model, "lgbm_model")

    print("✅ LightGBM model logged in MLflow")
    
    return model, mae  # Ensure the model and mae are returned

# STEP 3: Feature Reduction Model (Top 10 Features based on Importance)
def feature_reduction_model(df, model):
    # Get feature importance from the trained model
    feature_importance = model.feature_importances_

    # Create a DataFrame with feature names and their importance
    feature_names = [f'lag_{i}' for i in range(1, 29)]  # 28 lag features
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    })

    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    # Select top 10 features
    top_10_features = feature_importance_df['feature'].head(10).tolist()
    print(f"✅ Selected top 10 features: {top_10_features}")

    # STEP 3a: Train model with top 10 features
    X = df[top_10_features]
    y = df['trip_duration']

    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train LightGBM model
    reduced_model = lgb.LGBMRegressor()
    reduced_model.fit(X_train, y_train)

    # Predict and calculate MAE
    y_pred = reduced_model.predict(X_test)
    mae_reduced = mean_absolute_error(y_test, y_pred)
    print(f"Reduced Model MAE: {mae_reduced}")

    return reduced_model, mae_reduced

# STEP 3b: Log Feature Reduction Model in MLflow
def log_feature_reduction_model(df, base_model):
    # Start MLflow run
    with mlflow.start_run():
        # Create lag features
        df_with_lags = create_lag_features(df, days=28)

        # Train feature reduction model and get MAE
        reduced_model, mae_reduced = feature_reduction_model(df_with_lags, base_model)

        # Log model and metrics in MLflow
        mlflow.log_metric("mae_reduced", mae_reduced)
        mlflow.log_param("model", "LightGBM with Feature Reduction (top 10 features)")
        mlflow.sklearn.log_model(reduced_model, "lgbm_model_reduced")

    print("✅ Feature Reduction model logged in MLflow")

# Main function to load, preprocess, log baseline, and train models
def main():
    # Set up MLflow experiment (important step)
    mlflow.set_tracking_uri(f"https://dagshub.com/sravyajaogulamba/citibike-prediction.mlflow")
    mlflow.set_experiment("cbtpsc-experiments")
    print("✅ Connected to DagsHub MLflow Tracking")
    
    df = load_data(INPUT_CSV)
    top_stations = select_top_locations(df)
    cleaned_df = clean_and_preprocess_data(df, top_stations)
    save_data_to_csv(cleaned_df)
    store_data_in_hopsworks(cleaned_df, API_KEY, PROJECT_NAME, FEATURE_GROUP_NAME)
    
    # Log Baseline model
    log_baseline_model(cleaned_df)
    
    # Train and log LightGBM model
    model, mae = log_lgbm_model(cleaned_df)
    
    # Log Feature Reduction model
    log_feature_reduction_model(cleaned_df, model)
    
    print("✅ Pipeline finished!")

# Run everything
main()
