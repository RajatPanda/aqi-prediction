import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load actual and predicted data
df_actual = pd.read_csv("final_with_aqi.csv")
df_predicted = pd.read_csv("predicted_aqi_2022_grouped.csv")

# --- PREPROCESS ACTUAL DATA ---
df_actual['From Date'] = pd.to_datetime(df_actual['From Date'])
df_actual['year'] = df_actual['From Date'].dt.year
df_actual['month'] = df_actual['From Date'].dt.month
df_actual_2022 = df_actual[df_actual['year'] == 2022]

# Group by state and month for actual AQI
monthly_actual = df_actual_2022.groupby(['state', 'month'])['AQI'].mean().reset_index()

# --- PREPROCESS PREDICTED DATA ---
df_predicted['Date'] = pd.to_datetime(df_predicted['Date'])
df_predicted['month'] = df_predicted['Date'].dt.month

# Group by state and month for predicted AQI
monthly_predicted = df_predicted.groupby(['state', 'month'])['Predicted AQI'].mean().reset_index()

# --- MERGE AND PLOT ---
df_compare = pd.merge(monthly_actual, monthly_predicted, on=['state', 'month'], how='outer')

# Plot for each state
states = df_compare['state'].unique()

for state in states:
    df_state = df_compare[df_compare['state'] == state].copy()
    df_state = df_state.set_index('month').reindex(range(1, 13)).reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(df_state['month'], df_state['AQI'], label='Actual AQI', marker='o')
    plt.plot(df_state['month'], df_state['Predicted AQI'], label='Predicted AQI', marker='s')
    plt.title(f"{state} - Monthly AQI Comparison (2022)")
    plt.xlabel("Month")
    plt.ylabel("AQI")
    plt.xticks(range(1, 13))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
