"""
CMOS Dynamic Power Predictor — Final Project Suite
Importing: cmos_dataset.csv
Outputs: 5 Presentation-Ready Figures (PNG)
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
import os
import warnings
from scipy.stats import norm

warnings.filterwarnings('ignore')

# ── GitHub Dark Mode Palette ────────────────────────────────────────────────
BG   = '#0d1117'
SURF = '#161b22'
SURF2= '#21262d'
BORD = '#30363d'
CYAN = '#39d0f0'
GRNA = '#3fb950'
ORNG = '#f0883e'
PURP = '#bc8cff'
RED  = '#f85149'
TEXT = '#e6edf3'
MUT  = '#8b949e'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': SURF,
    'axes.edgecolor': BORD, 'axes.labelcolor': TEXT,
    'xtick.color': MUT, 'ytick.color': MUT,
    'text.color': TEXT, 'grid.color': BORD,
    'grid.linewidth': 0.5, 'font.family': 'monospace',
    'legend.facecolor': SURF2, 'legend.edgecolor': BORD,
})

# ── 1. Data Import ───────────────────────────────────────────────────────────
csv_path = 'cmos_dataset.csv'

if os.path.exists(csv_path):
    print(f"✓ Found {csv_path}. Importing data...")
    df = pd.read_csv(csv_path)
else:
    print(f"X Error: {csv_path} not found in the local directory!")
    exit()

# ── 2. Training Logic ────────────────────────────────────────────────────────
X = df[['switching_activity','load_capacitance_fF','supply_voltage_V','clock_frequency_GHz']].values
y = df['power_W'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Models
lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)

rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Metrics for Figures
r2_rf = r2_score(y_test, y_pred_rf)
r2_lr = r2_score(y_test, y_pred_lr)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
feat_names = ['Switching\nActivity (α)', 'Load Cap\n(fF)', 'Supply\nVoltage (V)', 'Clock Freq\n(GHz)']
importances = rf.feature_importances_


fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
fig.suptitle('CMOS Dynamic Power Predictor — Model Performance', fontsize=15, color=TEXT, fontweight='bold', y=1.01)

ax = axes[0]
ax.scatter(y_test*1e6, y_pred_rf*1e6, alpha=0.25, s=8, color=CYAN)
mn, mx = 0, max(y_test.max(), y_pred_rf.max())*1e6
ax.plot([mn,mx],[mn,mx], '--', color=GRNA, lw=1.8, label='Perfect fit')
ax.set_xlabel('Actual Power (µW)'); ax.set_ylabel('Predicted Power (µW)')
ax.set_title('Actual vs Predicted\n(Random Forest)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1]
res_uw = (y_test - y_pred_rf) * 1e6
ax.hist(res_uw, bins=60, color=PURP, alpha=0.7, density=True)
mu, sigma = res_uw.mean(), res_uw.std()
xs = np.linspace(res_uw.min(), res_uw.max(), 200)
ax.plot(xs, norm.pdf(xs, mu, sigma), color=ORNG, lw=2, label='Normal Fit')
ax.set_title('Prediction Error\nDistribution')
ax.set_xlabel('Residual (µW)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[2]
bars = ax.barh(feat_names, importances, color=[CYAN, PURP, ORNG, GRNA], height=0.55)
ax.set_title('Physical Driver\nImportance (RF)')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('plot1_main_performance.png', dpi=180, facecolor=BG)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
idx = np.argsort(y_test)[:200]
x_plot = np.arange(len(idx))

for ax, (preds, label, color, r2) in zip(axes, [(y_pred_lr[idx], 'Linear Regression', ORNG, r2_lr), (y_pred_rf[idx], 'Random Forest', CYAN, r2_rf)]):
    ax.fill_between(x_plot, y_test[idx]*1e6, preds*1e6, alpha=0.15, color=color)
    ax.plot(x_plot, y_test[idx]*1e6, color=TEXT, lw=1.2, label='Actual', alpha=0.7)
    ax.plot(x_plot, preds*1e6, color=color, lw=1.5, label=label)
    ax.set_title(f'{label} | R² = {r2:.4f}')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot2_model_comparison.png', dpi=180, facecolor=BG)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
corr = df.corr()
sns.heatmap(corr, ax=axes[0], annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, linecolor=BORD)
axes[0].set_title('Feature Correlation Matrix')

axes[1].hist(df['power_W']*1e6, bins=80, color=CYAN, alpha=0.7, log=True)
axes[1].set_title('Power Distribution (log scale)')
axes[1].set_xlabel('Power (µW)')
plt.savefig('plot3_data_analysis.png', dpi=180, facecolor=BG)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
v_vals = np.linspace(0.5, 3.3, 200)
p_v = 0.5 * 50e-15 * (v_vals**2) * 1e9
axes[0].plot(v_vals, p_v*1e6, color=CYAN, lw=2.5, label='P ∝ V²')
axes[0].set_title('Why V² Matters\n(Quadratic Relationship)')
axes[0].set_xlabel('Voltage (V)'); axes[0].set_ylabel('Power (µW)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

f_vals = np.linspace(1e6, 5e9, 200)
p_f = 0.5 * 50e-15 * (1.8**2) * f_vals
axes[1].plot(f_vals/1e9, p_f*1e6, color=GRNA, lw=2.5, label='P ∝ f')
axes[1].set_title('Linear Frequency Scaling')
axes[1].set_xlabel('Frequency (GHz)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.savefig('plot4_physics_verify.png', dpi=180, facecolor=BG)

fig = plt.figure(figsize=(16, 4), facecolor=BG)
ax = fig.add_axes([0,0,1,1])
ax.set_xlim(0,16); ax.set_ylim(0,4); ax.axis('off')

ax.text(8, 3.5, ' CMOS Dynamic Power Predictor', ha='center', fontsize=20, color=TEXT, fontweight='bold')
ax.axhline(2.85, color=BORD, lw=1, xmin=0.05, xmax=0.95)

metrics = [
    ('Dataset', f'{len(df)}', 'CSV samples'),
    ('LR R²', f'{r2_lr:.4f}', 'Linear Baseline'),
    ('RF R²', f'{r2_rf:.4f}', 'Best Model'),
    ('RF RMSE', f'{rmse_rf*1e6:.1f} µW', 'Prediction Error'),
]

for i, (label, val, sub) in enumerate(metrics):
    x = 2.5 + i*3.5
    ax.add_patch(plt.Rectangle((x-1.5, 0.5), 3.0, 2.0, facecolor=SURF, edgecolor=BORD))
    ax.text(x, 2.0, label, ha='center', color=MUT, fontsize=10)
    ax.text(x, 1.4, val, ha='center', color=CYAN if 'RF' in label else TEXT, fontsize=16, fontweight='bold')
    ax.text(x, 0.9, sub, ha='center', color=MUT, fontsize=8)

plt.savefig('plot5_summary_banner.png', dpi=180, facecolor=BG)
plt.show()

print("\n✓ Project Successful. All 5 figures saved to local folder.")
with open('results.json', 'w') as f:
    json.dump(results, f)

print(f"Linear Regression  → R²: {r2_lr:.4f}  RMSE: {np.sqrt(mse_lr):.6f} W")
print(f"Random Forest      → R²: {r2_rf:.4f}  RMSE: {np.sqrt(mse_rf):.6f} W")
print("Results saved to results.json")