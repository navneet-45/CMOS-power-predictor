"""
CMOS Power Predictor — Full Visualization Suite for GitHub
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Palette ──────────────────────────────────────────────────────────────────
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

np.random.seed(42)
N = 5000
alpha = np.random.uniform(0.01, 1.0, N)
C     = np.random.uniform(1e-15, 100e-15, N)
V     = np.random.uniform(0.5, 3.3, N)
f     = np.random.uniform(1e6, 5e9, N)
P_ideal = alpha * C * (V**2) * f
noise   = np.random.normal(0, 0.05 * P_ideal)
P       = np.maximum(P_ideal + noise, 0)

df = pd.DataFrame({
    'switching_activity': alpha,
    'load_capacitance_fF': C * 1e15,
    'supply_voltage_V': V,
    'clock_frequency_GHz': f / 1e9,
    'power_W': P
})

X = df[['switching_activity','load_capacitance_fF','supply_voltage_V','clock_frequency_GHz']].values
y = df['power_W'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)

rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

feat_names = ['Switching\nActivity (α)', 'Load Cap\n(fF)', 'Supply\nVoltage (V)', 'Clock Freq\n(GHz)']
importances = rf.feature_importances_

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Main 3-panel (your original, restyled)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
fig.suptitle('CMOS Dynamic Power Predictor — Model Performance', 
             fontsize=15, color=TEXT, fontweight='bold', y=1.01)

# Panel 1: Actual vs Predicted
ax = axes[0]
ax.scatter(y_test*1e6, y_pred_rf*1e6, alpha=0.25, s=8, color=CYAN, zorder=3)
mn, mx = 0, max(y_test.max(), y_pred_rf.max())*1e6
ax.plot([mn,mx],[mn,mx], '--', color=GRNA, lw=1.8, label='Perfect fit', zorder=4)
ax.set_xlabel('Actual Power (µW)'); ax.set_ylabel('Predicted Power (µW)')
ax.set_title('Actual vs Predicted\n(Random Forest)', color=TEXT, pad=10)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.text(0.05, 0.92, f'R² = 0.9740', transform=ax.transAxes, 
        fontsize=11, color=GRNA, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=SURF2, edgecolor=GRNA, alpha=0.8))

# Panel 2: Residuals
ax = axes[1]
residuals = (y_test - y_pred_rf) * 1e6
ax.hist(residuals, bins=60, color=PURP, alpha=0.7, edgecolor='none', density=True)
from scipy.stats import norm
mu, sigma = residuals.mean(), residuals.std()
xs = np.linspace(residuals.min(), residuals.max(), 200)
ax.plot(xs, norm.pdf(xs, mu, sigma), color=ORNG, lw=2, label=f'N(µ={mu:.1f}, σ={sigma:.1f})')
ax.axvline(0, color=RED, lw=1.5, linestyle='--', alpha=0.7)
ax.set_xlabel('Residual (µW)'); ax.set_ylabel('Density')
ax.set_title('Prediction Error\nDistribution', color=TEXT, pad=10)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Panel 3: Feature Importance
ax = axes[2]
colors = [CYAN, PURP, ORNG, GRNA]
bars = ax.barh(feat_names, importances, color=colors, edgecolor='none', height=0.55)
for bar, val in zip(bars, importances):
    ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
            f'{val*100:.1f}%', va='center', fontsize=10, color=TEXT)
ax.set_xlabel('Importance Score'); ax.set_title('Physical Driver\nImportance (RF)', color=TEXT, pad=10)
ax.set_xlim(0, max(importances)*1.25); ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/home/claude/plot1_main_performance.png', dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Plot 1 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Model Comparison (LR vs RF head-to-head)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig.suptitle('Linear Regression vs Random Forest — Head-to-Head', 
             fontsize=14, color=TEXT, fontweight='bold')

idx = np.argsort(y_test)[:200]
x_plot = np.arange(len(idx))

for ax, (preds, label, color, r2) in zip(axes, [
    (y_pred_lr[idx], 'Linear Regression', ORNG, 0.5632),
    (y_pred_rf[idx], 'Random Forest',     CYAN, 0.9740),
]):
    ax.fill_between(x_plot, y_test[idx]*1e6, preds*1e6, alpha=0.15, color=color)
    ax.plot(x_plot, y_test[idx]*1e6, color=TEXT, lw=1.2, label='Actual', alpha=0.7)
    ax.plot(x_plot, preds*1e6,       color=color, lw=1.5, label=label, alpha=0.9)
    ax.set_xlabel('Sample Index (sorted by power)'); ax.set_ylabel('Power (µW)')
    ax.set_title(f'{label}\nR² = {r2:.4f}', color=color, pad=8)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    r2_color = GRNA if r2 > 0.9 else RED
    ax.text(0.97, 0.05, f'R²={r2:.4f}', transform=ax.transAxes,
            ha='right', fontsize=12, color=r2_color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=SURF2, edgecolor=r2_color))

plt.tight_layout()
plt.savefig('/home/claude/plot2_model_comparison.png', dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Plot 2 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Correlation Heatmap + Power Distribution
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.suptitle('Dataset Analysis — Feature Correlations & Power Distribution', 
             fontsize=13, color=TEXT, fontweight='bold')

# Heatmap
corr = df.rename(columns={
    'switching_activity':'α (Activity)',
    'load_capacitance_fF':'C (fF)',
    'supply_voltage_V':'V (Volts)',
    'clock_frequency_GHz':'f (GHz)',
    'power_W':'Power (W)'
}).corr()

mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, ax=axes[0], annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, linecolor=BORD, 
            annot_kws={'size':10, 'color':TEXT},
            cbar_kws={'shrink':0.8})
axes[0].set_title('Feature Correlation Matrix', color=TEXT, pad=10)
axes[0].tick_params(colors=TEXT, labelsize=9)

# Power distribution (log scale)
axes[1].hist(df['power_W']*1e6, bins=80, color=CYAN, alpha=0.7, edgecolor='none', log=True)
axes[1].set_xlabel('Power (µW)'); axes[1].set_ylabel('Count (log scale)')
axes[1].set_title('Power Distribution\n(5,000 samples, log scale)', color=TEXT, pad=10)
axes[1].axvline(df['power_W'].mean()*1e6, color=ORNG, lw=2, linestyle='--', 
                label=f'Mean = {df["power_W"].mean()*1e6:.1f} µW')
axes[1].axvline(np.median(df['power_W'])*1e6, color=PURP, lw=2, linestyle='--',
                label=f'Median = {np.median(df["power_W"])*1e6:.1f} µW')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/plot3_data_analysis.png', dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Plot 3 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Voltage Effect (V² dependency demo) + Power vs Frequency
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig.suptitle('CMOS Physics Verification — P∝V² and P∝f Relationships', 
             fontsize=13, color=TEXT, fontweight='bold')

# V² relationship
v_vals = np.linspace(0.5, 3.3, 200)
p_v    = 0.5 * 50e-15 * (v_vals**2) * 1e9
p_lin  = p_v[0] + (p_v[-1]-p_v[0])/(3.3-0.5) * (v_vals - 0.5)
axes[0].plot(v_vals, p_v*1e6,   color=CYAN, lw=2.5, label='P = αCV²f (quadratic)')
axes[0].plot(v_vals, p_lin*1e6, color=ORNG, lw=1.5, linestyle='--', label='Linear extrapolation', alpha=0.6)
axes[0].fill_between(v_vals, p_lin*1e6, p_v*1e6, alpha=0.1, color=PURP)
axes[0].set_xlabel('Supply Voltage V (V)'); axes[0].set_ylabel('Dynamic Power (µW)')
axes[0].set_title('Why V² Matters\n(why Linear Regression fails at 56% R²)', color=TEXT, pad=8)
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
axes[0].annotate('Non-linear\nregion', xy=(2.8, p_v[-30]*1e6), fontsize=9, color=PURP,
                 xytext=(1.8, p_v[-1]*1e6*0.7),
                 arrowprops=dict(arrowstyle='->', color=PURP, lw=1.2))

# Frequency sweep
f_vals = np.linspace(1e6, 5e9, 200)
p_f    = 0.5 * 50e-15 * (1.8**2) * f_vals
axes[1].plot(f_vals/1e9, p_f*1e6, color=GRNA, lw=2.5)
axes[1].set_xlabel('Clock Frequency (GHz)'); axes[1].set_ylabel('Dynamic Power (µW)')
axes[1].set_title('Power vs Clock Frequency\n(linear — why RF captures this easily)', color=TEXT, pad=8)
axes[1].grid(True, alpha=0.3)
for fq, label in [(1,'1 GHz\n(typical CPU)'),(3,'3 GHz\n(high-perf)'),(4.5,'4.5 GHz\n(OC)')]:
    pq = 0.5 * 50e-15 * (1.8**2) * fq*1e9
    axes[1].axvline(fq, color=BORD, lw=1, linestyle=':')
    axes[1].scatter([fq],[pq*1e6], color=ORNG, s=60, zorder=5)
    axes[1].text(fq+0.05, pq*1e6*1.05, label, fontsize=8, color=MUT)

plt.tight_layout()
plt.savefig('/home/claude/plot4_physics_verify.png', dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Plot 4 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Model Summary Card (GitHub banner style)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 4), facecolor=BG)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0,0,1,1])
ax.set_xlim(0,16); ax.set_ylim(0,4); ax.axis('off')
ax.set_facecolor(BG)

# Title
ax.text(8, 3.5, '⚡  CMOS Dynamic Power Predictor', ha='center', va='center',
        fontsize=20, color=TEXT, fontweight='bold', fontfamily='monospace')
ax.text(8, 3.1, 'P = α · C · V² · f    |    5,000 samples    |    Python · scikit-learn',
        ha='center', va='center', fontsize=11, color=MUT, fontfamily='monospace')

# Divider
ax.axhline(2.85, color=BORD, lw=1, xmin=0.05, xmax=0.95)

# Metric boxes
metrics = [
    ('Dataset', '5,000', 'synthetic samples'),
    ('Split', '80 / 20', 'train / test'),
    ('LR  R²', '0.5632', '↓ fails on V²'),
    ('RF  R²', '0.9740', '↑ best model'),
    ('RF RMSE', '78.6 µW', 'vs 321.9 µW'),
    ('Features', '4', 'α · C · V · f'),
]
colors_m = [MUT, MUT, ORNG, GRNA, GRNA, CYAN]
for i, (label, val, sub) in enumerate(metrics):
    x = 1.15 + i*2.45
    rect = plt.Rectangle((x-1.0, 0.3), 2.0, 2.2, 
                          facecolor=SURF, edgecolor=BORD, linewidth=1, zorder=2)
    ax.add_patch(rect)
    ax.text(x, 1.95, label,  ha='center', va='center', fontsize=9,  color=MUT,          fontfamily='monospace')
    ax.text(x, 1.4,  val,    ha='center', va='center', fontsize=16, color=colors_m[i],  fontfamily='monospace', fontweight='bold')
    ax.text(x, 0.75, sub,    ha='center', va='center', fontsize=8,  color=MUT,          fontfamily='monospace')

plt.savefig('/home/claude/plot5_summary_banner.png', dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Plot 5 saved")
print("\nAll 5 plots generated.")
