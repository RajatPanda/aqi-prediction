import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load actual AQI data
df_actual = pd.read_csv("final_with_aqi.csv")
df_actual['From Date'] = pd.to_datetime(df_actual['From Date'])
df_actual['year'] = df_actual['From Date'].dt.year
df_actual = df_actual[df_actual['year'] == 2022]
df_actual['month'] = df_actual['From Date'].dt.month
monthly_actual = df_actual.groupby(['state', 'month'])['AQI'].mean().reset_index()

# Load predictions for both models
model_files = {
    'XGBoost': 'predicted_aqi_2022_xgb.csv',
    'Random Forest': 'predicted_aqi_2022_rf.csv'
}

for model_name, file_name in model_files.items():
    df_pred = pd.read_csv(file_name)
    df_pred['Date'] = pd.to_datetime(df_pred['Date'])
    df_pred['month'] = df_pred['Date'].dt.month

    monthly_pred = df_pred.groupby(['state', 'month'])['Predicted AQI'].mean().reset_index()

    # Merge and calculate error
    df_error = pd.merge(monthly_actual, monthly_pred, on=['state', 'month'], how='inner')
    df_error['Error'] = df_error['Predicted AQI'] - df_error['AQI']

    # Pivot for heatmap
    heatmap_data = df_error.pivot(index='state', columns='month', values='Error')

    # Plot
    plt.figure(figsize=(14, 8))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(
        heatmap_data, annot=True, fmt=".1f", cmap="coolwarm", center=0,
        linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Prediction Error'}
    )
    ax.set_title(f"{model_name} - Prediction Error Heatmap (2022)", fontsize=16)
    ax.set_xlabel("Month", fontsize=14)
    ax.set_ylabel("State", fontsize=14)
    plt.tight_layout()
    plt.show()
