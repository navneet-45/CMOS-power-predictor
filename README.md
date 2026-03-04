# CMOS Dynamic Power Consumption Predictor

> Machine Learning on VLSI Dynamic Power — Linear Regression vs Random Forest

---

## Problem Statement

In VLSI/CMOS circuit design, **dynamic power consumption** is a critical metric. Engineers need to estimate power quickly without running full SPICE simulations. This project builds an ML model to predict power from circuit parameters using the fundamental CMOS power equation.

## The Physics

```
P = α · C · V² · f
```

| Symbol | Parameter | Range Used |
|--------|-----------|------------|
| α | Switching activity (0–1) | 0.01 – 1.0 |
| C | Load capacitance | 1 – 100 fF |
| V | Supply voltage | 0.5 – 3.3 V |
| f | Clock frequency | 1 MHz – 5 GHz |

## Dataset

- **5,000 synthetic samples** generated using the CMOS equation
- ±5% Gaussian noise added to simulate real-world variation
- 80/20 train/test split

```python
P_ideal = alpha * C * (V**2) * f
noise   = np.random.normal(0, 0.05 * P_ideal)
P       = P_ideal + noise
```

## Results

| Model | R² Score | RMSE |
|-------|----------|------|
| Linear Regression | 0.5632 | 321.9 µW |
| **Random Forest** | **0.9740** | **78.6 µW** |

**Random Forest wins by a large margin.** Linear regression assumes a linear relationship, but power has a quadratic dependence on voltage (V²) and multiplicative interactions between features — RF captures this naturally.

## Feature Importance (RF)

All four features contribute almost equally (~25% each), confirming the balanced multiplicative structure of the CMOS equation.

```
Clock Freq (GHz):    25.8%
Switching Activity:  24.9%
Load Cap (fF):       24.9%
Supply Voltage (V):  24.3%
```

## Applications

- Pre-silicon power estimation in RTL design
- DVFS policy learning
- Power-aware synthesis and floorplanning
- Battery life estimation in mobile SoCs

---

