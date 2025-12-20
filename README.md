# Oil Price Impact on Fossil and Renewable Energy Sector Returns
### Data Science & Advanced Programming – Final Project  
**Author:** Luca Schwebin  
**Period:** 2018–2024  
**Assets studied:** WTI (CL=F), XLE (Fossil Energy ETF), ICLN (Renewable Energy ETF)

---

## 📌 Table of Contents
1. [Project Overview](#1-project-overview)  
2. [Repository Structure](#2-repository-structure)  
3. [How to Run the Project](#3-how-to-run-the-project)  
4. [Summary of Notebooks](#4-summary-of-notebooks)  
5. [Key Findings](#5-key-findings)  
6. [Limitations](#6-limitations)  
7. [Conclusion](#7-conclusion)  
8. [Author](#8-author)

---

## 1. Project Overview
This project examines whether **daily oil price returns (WTI)** affect the returns of the **fossil energy sector (XLE)** and the **renewable energy sector (ICLN)**.

The study focuses on:

- Statistical relationships between sector returns  
- Predictability using machine learning models  
- Robustness via time-series validation  
- Reaction to extreme oil shocks  

All data comes from **Yahoo Finance** and all outputs are generated automatically.

---

## 2. Repository Structure

```
oil-energy-project/
│
├── data/
│   ├── prices_2018_2024.parquet
│   ├── log_returns_2018_2024.parquet
│   ├── model_features_2008_2024.parquet
│   ├── y_pred_linreg.parquet
│   ├── y_pred_rf.parquet
│   └── y_test_targets.parquet
│
├── notebooks/
│   ├── 01_data_download.ipynb
│   ├── 02_log_returns.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_modeling_final.ipynb
│   ├── 06_time_series_validation.ipynb
│   ├── 07_oil_shock_analysis.ipynb
│   └── 08_final_economic_analysis.ipynb
│
├── results/
│   ├── model_performance_metrics.csv
│   ├── model_performance_metrics.parquet
│   ├── cv_results_timeseries.csv
│   ├── cv_summary_timeseries.csv
│   ├── oil_shock_reaction_summary.csv
│   ├── r2_timeseries_XLE_ret.png
│   ├── r2_timeseries_ICLN_ret.png
│   ├── shock_timeline.png
│   └── shock_average_reaction.png
│
└── README.md
```

---

## 3. How to Run the Project

Run the notebooks **IN ORDER**:

➡️ **01 → 02 → 03 → 04 → 05 → 06 → 07 → 08**

All datasets and outputs are generated automatically.

---

## 4. Summary of Notebooks

### **01 – Data Download**  
Downloads Yahoo Finance prices for WTI, XLE, ICLN and stores them as parquet files.

### **02 – Log Returns**  
Cleans prices and computes daily log returns.

### **03 – Feature Engineering**  
Generates lag features and rolling statistics for ML modeling.

### **04 – Modeling**  
Models tested:  
- Naive  
- Linear Regression  
- Random Forest  
Metrics: RMSE, MAE, R².

### **05 – Final Modeling**  
Retrains the best model and exports predictions.

### **06 – Time-Series Validation**  
Applies walk-forward **TimeSeriesSplit** validation.

### **07 – Oil Shock Analysis**  
Detects oil shocks using the **5% and 95% quantiles**, evaluates reactions of XLE and ICLN.

### **08 – Final Economic Summary**  
Summarizes results and gives economic interpretation.

---

## 5. Key Findings

### **Predictability**
- **XLE** shows **moderate predictability**.  
- **ICLN** is **much harder to predict**, suggesting weaker sensitivity to oil dynamics.

### **Model Performance**
- **Random Forest performs best**.  
- **Naive model performs worst**.

### **Time-Series Stability**
- Performance fluctuates with changing market conditions (expected in financial data).

### **Reaction to Oil Shocks**
- **XLE reacts strongly and positively** to oil shocks.  
- **ICLN reacts weakly**, confirming partial insulation from oil movements.

---

## 6. Limitations
- Daily data only  
- No macroeconomic variables  
- All features derived from WTI  
- Shock window limited to ±3 days  

---

## 7. Conclusion
The **fossil sector** (XLE) is significantly influenced by oil price movements, while the **renewable sector (ICLN)** is more insulated.  
Machine learning models detect these relationships, but **predictability remains limited** due to financial noise.

---

## 8. Author
**Luca Schwebin**  
HEC Lausanne – Data Science & Advanced Programming  
**2025–2026**