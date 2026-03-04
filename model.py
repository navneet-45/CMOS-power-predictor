"""
CMOS Dynamic Power Consumption Predictor
P = alpha * C * V^2 * f
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import json, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
N = 5000

alpha = np.random.uniform(0.01, 1.0, N)       # switching activity
C     = np.random.uniform(1e-15, 100e-15, N)   # load capacitance (F)
V     = np.random.uniform(0.5, 3.3, N)         # supply voltage (V)
f     = np.random.uniform(1e6, 5e9, N)         # clock frequency (Hz)

P_ideal = alpha * C * (V**2) * f               # CMOS power equation
noise   = np.random.normal(0, 0.05 * P_ideal)  # 5% noise
P       = np.maximum(P_ideal + noise, 0)        # watts

df = pd.DataFrame({
    'switching_activity': alpha,
    'load_capacitance_fF': C * 1e15,
    'supply_voltage_V': V,
    'clock_frequency_GHz': f / 1e9,
    'power_W': P
})

df.to_csv('cmos_dataset.csv', index=False)

# --- Features ---
X = df[['switching_activity','load_capacitance_fF','supply_voltage_V','clock_frequency_GHz']].values
y = df['power_W'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# --- Linear Regression ---
lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr  = r2_score(y_test, y_pred_lr)

# --- Random Forest ---
rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf  = r2_score(y_test, y_pred_rf)

feat_names = ['Switching Activity (α)', 'Load Cap (fF)', 'Supply Voltage (V)', 'Clock Freq (GHz)']
importances = rf.feature_importances_.tolist()

# --- Sample predictions ---
sample_idx = np.random.choice(len(y_test), 200, replace=False)
actual_sample    = y_test[sample_idx].tolist()
pred_lr_sample   = y_pred_lr[sample_idx].tolist()
pred_rf_sample   = y_pred_rf[sample_idx].tolist()

results = {
    "linear_regression": {"mse": mse_lr, "r2": r2_lr, "rmse": np.sqrt(mse_lr)},
    "random_forest":     {"mse": mse_rf, "r2": r2_rf, "rmse": np.sqrt(mse_rf)},
    "feature_importances": dict(zip(feat_names, importances)),
    "actual":   actual_sample,
    "pred_lr":  pred_lr_sample,
    "pred_rf":  pred_rf_sample,
    "n_train": len(X_train),
    "n_test":  len(X_test)
}

with open('results.json', 'w') as f:
    json.dump(results, f)

print(f"Linear Regression  → R²: {r2_lr:.4f}  RMSE: {np.sqrt(mse_lr):.6f} W")
print(f"Random Forest      → R²: {r2_rf:.4f}  RMSE: {np.sqrt(mse_rf):.6f} W")
print("Results saved to results.json")