# Project Proposal

## Title: Impact of Oil Prices on the Stock Performance of Fossil and Renewable Energy Sectors

The original PDF version of this proposal is available here: `docs/project_proposal.pdf`.

This project examines how fluctuations in crude oil prices affect the stock performance of two energy sectors: fossil fuels (XLE ETF) and renewable energy (ICLN ETF). The goal is to determine whether renewable energy stocks react differently to oil price shocks and whether they demonstrate a growing independence from fossil fuel markets.

Daily historical data will be retrieved from Yahoo Finance for the period 2018–2024 (approximately 1,500 trading days). Prices will be transformed into logarithmic daily returns, and the dataset will be enhanced with lagged features (t-1, t-2, t-5) and rolling-window statistics (e.g. 5-day moving average of oil returns) to capture delayed effects rather than only contemporaneous correlations.

Data will be split using a chronological train/test split (80/20), and a walk-forward time-series validation will be used to assess performance stability. The baseline will be a naive forecast ("tomorrow’s return = today’s return"). The project will compare two models:
1. Multiple linear regression (interpretable benchmark)
2. Random Forest regressor (captures nonlinear relationships)

Success will be measured using RMSE, MAE, and R² on the test set.

Deliverables will include time-series visualizations, a feature-importance analysis, model comparison tables, and forecast plots. The final output will quantify the sensitivity of each sector to oil price movements and evaluate whether renewable energy exhibits growing independence from fossil fuel dynamics.