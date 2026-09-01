# 🩸 Blood Glucose Forecasting for Type 1 Diabetes — BrisT1D (Kaggle)

*[Leggi in italiano](README_diabet.md)*

A comparative forecasting study — classical statistical models, decomposition-based models, tree ensembles, and a recurrent neural network — for predicting future blood glucose in a Type 1 diabetes patient, using real CGM, insulin pump, and smartwatch data.

## 🎯 Objective

Reconstruct a usable time series from a clinical dataset distributed in a non-standard format, and systematically compare five forecasting approaches — **SARIMA**, **SARIMAX**, **Prophet**, **Gradient Boosting**, and **LSTM** — to predict a patient's blood glucose one hour ahead, rigorously distinguishing between a **blind forecast** scenario (no updates) and a **walk-forward** scenario (hourly updates, as in a real-time monitoring system).

## 📁 Dataset

- **Source:** [BrisT1D Blood Glucose Prediction](https://www.kaggle.com/competitions/brist1d) (Kaggle)
- **Content:** real data collected via CGM, insulin pump, and smartwatch from young UK adults with Type 1 Diabetes, sampled every 5 minutes
- **Main challenge:** the dataset is not distributed as a time series but in "wide" format designed for tabular regression — the whole project starts by reconstructing a continuous time series for a single patient (`p01`)
- **Ethical considerations:** sensitive health data (GDPR Art. 9), pseudonymized; used strictly for educational/research purposes, no real clinical application

## 🔬 Methodology

**1. Time series reconstruction**
Extraction of the "current" glucose reading (`bg-0:00`) for each row of patient `p01`, reconstruction of a relative time axis (the dataset provides no absolute dates for privacy reasons), resampling to 15 minutes → constrained linear interpolation (`limit=2`, max 30 min) → hourly downsampling, with explicit justification of each parameter choice against the biological plausibility of the signal.

**2. Exploratory analysis and Time in Range (TIR)**
Time series visualization, standard clinical metrics (Time in Range, Time Below/Above Range per ADA/ISPAD guidelines), hourly boxplots to identify the circadian pattern (dawn phenomenon).

**3. STL decomposition, stationarity, ACF/PACF**
Trend/Seasonality/Residual decomposition (period=24, robust=True), ADF + KPSS stationarity tests, ACF/PACF analysis to identify AR/MA orders.

**4. Classical statistical modeling — SARIMA and SARIMAX**
Systematic comparison between manually identified orders and `auto_arima`, model selection via AIC and Likelihood Ratio Test, residual diagnostics (Ljung-Box). **Cross-Correlation Function (CCF)** analysis to identify the true causal lag of insulin's glucose-lowering effect (2-3h, consistent with real pharmacokinetics) before including it in SARIMAX.

**5. Prophet**
Additive decomposition with automatic changepoints and Fourier-series seasonality, with and without exogenous regressors.

**6. Gradient Boosting + SHAP interpretability**
Explicit feature engineering (autoregressive lags, seasonal lags, delayed insulin, cyclical hour encoding), identification and correction of a **data leakage** case, SHAP-based interpretation, bias correction on extreme events via **weighted loss** and **Quantile Regression** for prediction intervals.

**7. LSTM**
Compact recurrent network (16 units, dropout, early stopping), with an explicit discussion of overfitting risk given the limited data available for a single patient.

**8. Final evaluation**
Retraining of the selected model on train+validation, walk-forward forecast on the Kaggle test set, with separate analysis of genuinely observed vs. interpolated hours to avoid an inflated evaluation.

## 📈 Results

**Group A — Blind forecast (168h, no updates)**

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| **Prophet** 🏆 | **2.674** | **3.383** | **39.3%** |
| Prophet + insulin/carbs | 2.734 | 3.429 | 40.3% |
| SARIMA | 2.898 | 3.496 | 44.8% |
| SARIMAX + insulin/carbs | 2.996 | 3.594 | 46.0% |

**Group B — Walk-forward (1h ahead, hourly updates)**

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| **SARIMA** 🏆 | **1.666** | **2.342** | **22.7%** |
| LSTM | 1.724 | 2.247 | 25.3% |
| Gradient Boosting | 1.734 | 2.300 | 25.5% |

**Final evaluation on the Kaggle test set** (SARIMA walk-forward): MAE = 0.422 on the full test set; MAE = 0.508 on hours genuinely observed by the sensor (429 out of 2,869 total points, the rest being interpolation due to the test set's low sampling density).

## 💡 Key Insights

- **Blind vs. walk-forward completely changes the model ranking**: Prophet wins the long-horizon forecast with no updates, but SARIMA wins in the realistic continuous-monitoring scenario — much of the apparent advantage of more flexible models shrinks drastically once the comparison is put on the same informational footing.
- **Exogenous variables (insulin, carbs) are statistically significant but do not improve out-of-sample forecasting** — a result confirmed independently by two model families (SARIMAX and Prophet+regressors), both of which got worse than their univariate versions.
- **CCF analysis reveals insulin's true causal lag** (2-3 hours, not 0 as initially assumed), consistent with the real pharmacokinetics of rapid-acting insulins — a concrete example of how a lag-0 correlation can reflect reverse causality (high glucose → insulin administered in response) rather than the true pharmacological effect.
- **All models systematically underestimate extreme events** (severe hyper/hypoglycemia), the most clinically relevant limitation of the project; a weighted loss and quantile regression mitigate but don't eliminate the issue.
- **A data leakage case identified and fixed** via anomalous feature importance (Gradient Boosting), explicitly documented in the notebook as part of the process rather than hidden.
- **LSTM does not beat a well-specified SARIMA** with ~2,000 observations — consistent with the theoretical expectation that recurrent networks need much larger data volumes to show a real advantage.

## ⚠️ Limitations

- Analysis calibrated on a single patient (`p01`); not automatically generalizable to other individuals
- Single train/validation split (7 days); a multi-window rolling-origin validation would be more robust
- Residual systematic bias on extreme events, only partially corrected

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `statsmodels` · `pmdarima` · `Prophet` · `LightGBM` · `SHAP` · `TensorFlow/Keras` · `scikit-learn` · `matplotlib` · `seaborn`

## 📂 Repository Structure

```
├── notebook ENG/
│   └── Diabet_Analysis ENG.ipynb
│   └── Diabet_Analysis ENG.pdf
├── data/
│   └── (BrisT1D dataset — see Dataset section)
├── requirements.txt
└── README.md
```

## ▶️ How to Run

```bash
git clone https://github.com/elenacascone/<Blood-Glucose-Forecasting>.git
cd <Blood-Glucose-Forecasting>
pip install -r requirements.txt
jupyter notebook notebook/Diabet_Analysis.ipynb
```

## 👤 Author

**Elena Cascone**
[LinkedIn](https://www.linkedin.com/in/elena-cascone-18ec/) · [GitHub](https://github.com/elenacascone)
