# Oil–Energy Relationship: Modeling, Validation, and Shock Analysis

## Project overview
This project studies the relationship between oil prices and energy-related financial assets, with a focus on fossil fuel and renewable energy markets.  
Using daily financial data, the project combines time-series modeling, walk-forward validation, and an event-based analysis of extreme oil price movements.

The analysis aims to assess predictive performance and to explore whether renewable-related assets exhibit different dynamics compared to fossil fuel assets when oil prices experience extreme variations.

---

## Data
Daily price data are obtained from Yahoo Finance for the period 2018–2024:
- WTI crude oil futures (CL=F)
- Energy Select Sector SPDR Fund (XLE)
- iShares Global Clean Energy ETF (ICLN)

Prices are aligned on common trading dates and transformed into log-returns for modeling purposes.

---

## Project structure

oil-energy-project/  
├── data/  
│   ├── log_returns_2018_2024.parquet  
│   ├── model_features_2018_2024.parquet  
│   └── prices_2018_2024.parquet  
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
├── outputs/  
│   ├── results/  
│   ├── plots/  
│   └── executed_08_final_economic_analysis.ipynb # Executed version 
│  
├── main.py  
├── environment.yml  
├── requirements.txt  
└── README.md  

---

## Notebook description

### 01_data_download.ipynb
Downloads daily price data and aligns all assets on common dates.  
The cleaned price panel is saved for subsequent analysis.

### 02_log_returns.ipynb
Computes daily log-returns from price data and handles non-positive prices appropriately.  
The resulting return series are saved for feature construction.

### 03_feature_engineering.ipynb
Constructs lagged returns and rolling statistics.  
Defines the prediction target as the next-day return.

### 04_modeling.ipynb
Implements baseline and predictive models using a chronological train/test split.  
Models include a linear regression and a random forest.

### 05_modeling_final.ipynb
Evaluates model performance on the test set and produces visual diagnostics.  
Performance metrics and plots are saved to disk.

### 06_time_series_validation.ipynb
Performs walk-forward validation using time-series splits to assess model stability over time.

### 07_oil_shock_analysis.ipynb
Identifies extreme oil price movements using tail events in WTI returns.  
Analyzes the average reaction of energy-related assets around these events.

### 08_final_economic_analysis.ipynb
Synthesizes modeling and shock analysis results and provides an economic interpretation of the findings.

---

## Methodological notes
- All train/test splits respect the temporal ordering of the data.
- The baseline model predicts a constant zero return.
- Extreme oil price movements are defined using empirical quantiles of oil returns and are used as a proxy for shocks.

---

## Main findings
- Predictive performance remains limited, consistent with the difficulty of forecasting daily financial returns.
- Fossil fuel and renewable-related assets exhibit heterogeneous reactions to extreme oil price movements.
- The results suggest weaker dependence between oil prices and renewable-related assets rather than full structural independence.

All numerical outputs are saved to the outputs/results directory, and figures are saved to the outputs/plots directory.

---

## Requirements
- Python 3.11  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- pyarrow  
- nbformat  
- nbconvert  
- yfinance  

---

## Environment setup
conda env create -f environment.yml  
conda activate oil-energy-project  
python main.py  

---

## Author
Luca Schweblin
