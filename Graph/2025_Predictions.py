import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and prepare data
df_2025 = pd.read_csv("predicted_aqi_2025_grouped.csv")
df_2025['Date'] = pd.to_datetime(df_2025['Date'])
df_2025['month'] = df_2025['Date'].dt.month

# ✅ Remove incorrect state
df_2025 = df_2025[df_2025['state'] != 'AS']

# ===============================
# 1. Heatmap: State vs Month AQI
# ===============================
monthly_state_avg = df_2025.groupby(['state', 'month'])['Predicted AQI'].mean().reset_index()
heatmap_data = monthly_state_avg.pivot(index='state', columns='month', values='Predicted AQI')

plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5, cbar_kws={'label': 'AQI'})
plt.title("State-wise Monthly Average Predicted AQI (2025)", fontsize=16)
plt.xlabel("Month")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# =====================================
# 2. Std. Deviation (Volatility) Plot
# =====================================
heatmap_data['AQI StdDev'] = heatmap_data.std(axis=1)
top_volatility = heatmap_data['AQI StdDev'].sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
top_volatility.plot(kind='bar', color='orange')
plt.title("Top 10 Most Volatile States (Monthly AQI Std. Dev in 2025)")
plt.ylabel("AQI Std. Deviation")
plt.xlabel("State")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# ===================================
# 3. Most Polluted Month per State
# ===================================
idx = monthly_state_avg.groupby('state')['Predicted AQI'].idxmax()
peak_months = monthly_state_avg.loc[idx].sort_values(by='Predicted AQI', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=peak_months, x='state', y='Predicted AQI', hue='month', palette='viridis')
plt.title("Most Polluted Month by State (2025)")
plt.ylabel("Peak Monthly AQI")
plt.xlabel("State")
plt.legend(title='Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define AQI categories with readable labels
def classify_aqi_labeled(aqi):
    if aqi <= 50:
        return 'Good (0–50)'
    elif aqi <= 100:
        return 'Satisfactory (51–100)'
    elif aqi <= 200:
        return 'Moderate (101–200)'
    elif aqi <= 300:
        return 'Poor (201–300)'
    elif aqi <= 400:
        return 'Very Poor (301–400)'
    else:
        return 'Severe (401–500)'

# Load data
df_2025 = pd.read_csv("predicted_aqi_2025_grouped.csv")
df_2025['AQI Category'] = df_2025['Predicted AQI'].apply(classify_aqi_labeled)

# Remove AS if still in data
df_2025 = df_2025[df_2025['state'] != 'AS']

# Count categories
category_counts = df_2025['AQI Category'].value_counts().sort_index()

# Plot
plt.figure(figsize=(8, 8))
category_counts.plot(
    kind='pie',
    autopct='%1.1f%%',
    colors=sns.color_palette('Set2'),
    textprops={'fontsize': 12}
)
plt.title("Predicted AQI Category Distribution (2025)", fontsize=14)
plt.ylabel("")
plt.tight_layout()
plt.show()
