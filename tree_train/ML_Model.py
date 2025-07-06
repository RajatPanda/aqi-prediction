import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
from datetime import datetime
from glob import glob
import os
import warnings
warnings.filterwarnings('ignore')

# AQI breakpoints
breakpoints = {
    'PM2.5 (ug/m3)': [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
                      (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)],
    'PM10 (ug/m3)': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
                     (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)],
    'NO2 (ug/m3)': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
                    (181, 280, 201, 300), (281, 400, 301, 400), (401, 600, 401, 500)],
    'SO2 (ug/m3)': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
                    (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2000, 401, 500)],
    'CO (mg/m3)': [(0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200),
                   (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 50, 401, 500)],
    'Ozone (ug/m3)': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
                      (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)],
    'NH3 (ug/m3)': [(0, 200, 0, 50), (201, 400, 51, 100), (401, 800, 101, 200),
                    (801, 1200, 201, 300), (1201, 1800, 301, 400), (1801, 2000, 401, 500)]
}

def calculate_subindex(value, bps):
    for (c_low, c_high, i_low, i_high) in bps:
        if c_low <= value <= c_high:
            return i_low + ((value - c_low) / (c_high - c_low)) * (i_high - i_low)
    return np.nan

def compute_aqi(df):
    print("📊 Calculating AQI...")
    pollutants = list(breakpoints.keys())
    df = df.set_index('From Date')
    df[pollutants] = df[pollutants].interpolate(method='time', limit_direction='forward').fillna(method='ffill')
    df = df.reset_index()
    sub_df = {col: df[col].apply(lambda x: calculate_subindex(x, breakpoints[col])) for col in pollutants if col in df.columns}
    df['AQI'] = pd.DataFrame(sub_df).max(axis=1, skipna=True)
    df['Dominant Pollutant'] = pd.DataFrame(sub_df).idxmax(axis=1, skipna=True)
    return df

def add_features(df):
    df['hour'] = df['From Date'].dt.hour
    df['day'] = df['From Date'].dt.day
    df['month'] = df['From Date'].dt.month
    df['weekday'] = df['From Date'].dt.weekday
    df['year'] = df['From Date'].dt.year
    return df

def process_and_save_each_file(folder_path, output_dir="processed"):
    os.makedirs(output_dir, exist_ok=True)
    all_files = glob(os.path.join(folder_path, '*.csv'))
    for file in all_files:
        try:
            print(f"📂 Processing: {file}")
            state_code = os.path.basename(file)[:2].upper()
            df = pd.read_csv(file)
            df.columns = [col.strip() for col in df.columns]
            df['From Date'] = pd.to_datetime(df['From Date'], errors='coerce')
            df = df.dropna(subset=['From Date']).sort_values('From Date')
            df['state'] = state_code
            useful_cols = ['From Date'] + list(breakpoints.keys()) + [
                'Temp (degree C)', 'RH (%)', 'WS (m/s)', 'WD (degree)', 'state']
            df = df[[col for col in useful_cols if col in df.columns]]
            df = compute_aqi(df)
            df = add_features(df)
            for col in df.select_dtypes(include='float'):
                df[col] = pd.to_numeric(df[col], downcast='float')
            df.to_csv(os.path.join(output_dir, os.path.basename(file)), index=False)
        except Exception as e:
            print(f"❌ Failed: {file} — {e}")

def load_combined_processed_data(folder="processed"):
    files = glob(os.path.join(folder, '*.csv'))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

def train_model(df, model_type='xgb'):
    print(f"🧠 Training model: {model_type.upper()}")
    full_features = [
        'PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO2 (ug/m3)', 'SO2 (ug/m3)',
        'CO (mg/m3)', 'Ozone (ug/m3)', 'NH3 (ug/m3)',
        'Temp (degree C)', 'RH (%)', 'WS (m/s)',
        'hour', 'day', 'month', 'weekday', 'year']
    feature_cols = [col for col in full_features if col in df.columns]
    df = df.dropna(subset=['AQI'] + feature_cols)
    X = df[feature_cols]
    y = df['AQI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=50, max_depth=8, n_jobs=-1, verbose=1, random_state=42)
    else:
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    print(f"✅ Model Trained ({model_type.upper()}): RMSE = {rmse:.2f}, R² = {r2:.2f}")
    joblib.dump(model, f"aqi_model_{model_type}.pkl")
    joblib.dump(feature_cols, f"model_features_{model_type}.pkl")
    return model, df

def predict_aqi_for_year(year=2022, model_type='xgb'):
    print(f"📅 Predicting AQI for {year} using {model_type.upper()}")
    df = pd.read_csv("final_with_aqi.csv")
    df['From Date'] = pd.to_datetime(df['From Date'])
    feature_cols = joblib.load(f"model_features_{model_type}.pkl")
    model = joblib.load(f"aqi_model_{model_type}.pkl")

    states = df['state'].unique()
    predictions = []

    for state in states:
        df_state = df[df['state'] == state]
        for date in pd.date_range(f"{year}-01-01", f"{year}-12-31"):
            avg_features = df_state[(df_state['day'] == date.day) & (df_state['month'] == date.month)].mean(numeric_only=True)
            input_data = {
                'hour': 12, 'day': date.day, 'month': date.month,
                'weekday': date.weekday(), 'year': date.year
            }
            for col in feature_cols:
                if col not in input_data:
                    input_data[col] = avg_features.get(col, 0)
            X_input = pd.DataFrame([input_data])[feature_cols]
            predicted_aqi = model.predict(X_input)[0]
            predictions.append({
                'Date': date.date(),
                'Predicted AQI': predicted_aqi,
                'state': state,
                'year': date.year,
                'weekday': date.weekday(),
                'day': date.day
            })

    result_df = pd.DataFrame(predictions)
    result_df.to_csv(f"predicted_aqi_{year}_{model_type}.csv", index=False)
    print(f"✅ Saved to predicted_aqi_{year}_{model_type}.csv")

# Main execution
if __name__ == "__main__":
    process_and_save_each_file("/Dataset AQI")  # Replace with your dataset path
    combined_df = load_combined_processed_data()
    for model_type in ['xgb', 'rf']:
        model, df_final = train_model(combined_df, model_type=model_type)
        if model_type == 'xgb':
            df_final.to_csv("final_with_aqi.csv", index=False)
        predict_aqi_for_year(2022, model_type=model_type)
