"""
CMOS Dynamic Power Consumption Predictor with Visualizations
Equation: P = alpha * C * V^2 * f
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings

warnings.filterwarnings('ignore')

# --- 1. Data Generation ---
np.random.seed(42)
N = 5000

alpha = np.random.uniform(0.01, 1.0, N)       # switching activity
C     = np.random.uniform(1e-15, 100e-15, N)   # load capacitance (F)
V     = np.random.uniform(0.5, 3.3, N)         # supply voltage (V)
f     = np.random.uniform(1e6, 5e9, N)         # clock frequency (Hz)

P_ideal = alpha * C * (V**2) * f               # Physics-based CMOS power equation
noise   = np.random.normal(0, 0.05 * P_ideal)  # Adding 5% measurement noise
P       = np.maximum(P_ideal + noise, 0)        # Final power in Watts

df = pd.DataFrame({
    'switching_activity': alpha,
    'load_capacitance_fF': C * 1e15,
    'supply_voltage_V': V,
    'clock_frequency_GHz': f / 1e9,
    'power_W': P
})

# Save dataset to the local folder
df.to_csv('cmos_dataset.csv', index=False)

# --- 2. Feature Engineering & Splitting ---
X = df[['switching_activity','load_capacitance_fF','supply_voltage_V','clock_frequency_GHz']].values
y = df['power_W'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# --- 3. Model Training ---
# Linear Regression
lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)

# Random Forest (Usually better for the non-linear V^2 relationship)
rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# --- 4. Metrics & Results Storage ---
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf  = r2_score(y_test, y_pred_rf)

feat_names = ['Switching Activity (α)', 'Load Cap (fF)', 'Supply Voltage (V)', 'Clock Freq (GHz)']
importances = dict(zip(feat_names, rf.feature_importances_))

results = {
    "linear_regression": {"r2": r2_score(y_test, y_pred_lr), "rmse": np.sqrt(mean_squared_error(y_test, y_pred_lr))},
    "random_forest":     {"r2": r2_rf, "rmse": np.sqrt(mse_rf)},
    "feature_importances": importances
}

with open('results.json', 'w') as f_json:
    json.dump(results, f_json, indent=4)

# --- 5. Visualizations for Presentation ---
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Actual vs Predicted Scatter Plot
sns.scatterplot(x=y_test, y=y_pred_rf, ax=axes[0], alpha=0.4, color='teal')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_title('Actual vs. Predicted Power (Random Forest)')
axes[0].set_xlabel('Actual Power (W)')
axes[0].set_ylabel('Predicted Power (W)')

# Plot 2: Residuals (Error) Distribution
residuals = y_test - y_pred_rf
sns.histplot(residuals, kde=True, ax=axes[1], color='m')
axes[1].set_title('Distribution of Prediction Errors')
axes[1].set_xlabel('Error (Watts)')

# Plot 3: Feature Importance (ECE physical drivers)
sns.barplot(x=list(importances.values()), y=list(importances.keys()), ax=axes[2], palette='viridis')
axes[2].set_title('Physical Drivers Importance')
axes[2].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('cmos_performance_analysis.png', dpi=300) # Save high-res image for project report
plt.show()

print(f"Random Forest Performance → R²: {r2_rf:.4f} | RMSE: {np.sqrt(mse_rf):.6f} W")
print("Project files (CSV, JSON, and PNG plots) saved to your local folder.")