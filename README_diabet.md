# Previsione della Glicemia in Pazienti con Diabete di Tipo 1 — BrisT1D (Kaggle)

*[Read this in English](README.en.md)*

Studio comparativo di modelli di forecasting — statistici classici, a decomposizione, ad alberi e a rete neurale ricorrente — per la previsione della glicemia futura di un paziente con diabete di tipo 1, a partire da dati reali di sensore CGM, microinfusore e smartwatch.

## Obiettivo

Ricostruire una serie storica utilizzabile a partire da un dataset clinico distribuito in formato non standard, e confrontare sistematicamente cinque approcci di previsione — **SARIMA**, **SARIMAX**, **Prophet**, **Gradient Boosting** e **LSTM** — per prevedere la glicemia di un paziente un'ora nel futuro, distinguendo rigorosamente tra scenario di **forecast cieco** (nessun aggiornamento) e scenario **walk-forward** (aggiornamento orario, come un vero sistema di monitoraggio in tempo reale).

## Dataset

- **Fonte:** [BrisT1D Blood Glucose Prediction](https://www.kaggle.com/competitions/brist1d) (Kaggle)
- **Contenuto:** dati reali raccolti da CGM, microinfusore di insulina e smartwatch su giovani adulti nel Regno Unito con Diabete di Tipo 1, campionati ogni 5 minuti
- **Sfida principale:** il dataset non è distribuito come serie storica, ma in formato "wide" pensato per la regressione tabellare — l'intero progetto parte dalla ricostruzione di una serie storica continua per un singolo paziente (`p01`)
- **Considerazioni etiche:** dati sanitari sensibili (GDPR art. 9), pseudonimizzati; uso esclusivamente didattico/di ricerca, nessun impiego clinico reale

## Metodologia

**1. Ricostruzione della serie storica**
Estrazione della glicemia "presente" (`bg-0:00`) di ogni riga per il paziente `p01`, ricostruzione di un asse temporale relativo (il dataset non fornisce date assolute per privacy), resampling a 15 minuti → interpolazione lineare vincolata (`limit=2`, max 30 min) → downsampling orario, con giustificazione esplicita della scelta di ogni parametro rispetto alla plausibilità biologica del segnale.

**2. Analisi esplorativa e Time in Range (TIR)**
Visualizzazione della serie, calcolo delle metriche cliniche standard (Time in Range, Time Below/Above Range secondo le linee guida ADA/ISPAD), boxplot orario per l'identificazione del pattern circadiano (dawn phenomenon).

**3. Decomposizione STL, stazionarietà, ACF/PACF**
Decomposizione Trend/Stagionalità/Residuo (period=24, robust=True), test di stazionarietà ADF + KPSS, analisi ACF/PACF per l'identificazione degli ordini AR/MA.

**4. Modellazione statistica classica — SARIMA e SARIMAX**
Confronto sistematico tra ordini identificati "ad occhio" e tramite `auto_arima`, selezione via AIC e Likelihood Ratio Test, verifica dei residui (Ljung-Box). Analisi di **Cross-Correlation Function (CCF)** per identificare il vero lag causale dell'effetto ipoglicemizzante dell'insulina (2-3h, coerente con la farmacocinetica reale) prima di includerla in SARIMAX.

**5. Prophet**
Decomposizione additiva con changepoint automatici e stagionalità via serie di Fourier, con e senza regressori esogeni.

**6. Gradient Boosting + interpretabilità SHAP**
Feature engineering esplicito (lag autoregressivi, lag stagionali, insulina ritardata, codifica ciclica dell'ora), identificazione e correzione di un caso di **data leakage**, interpretazione tramite valori SHAP, correzione del bias sugli eventi estremi tramite **loss pesata** e **Quantile Regression** per intervalli di previsione.

**7. LSTM**
Rete ricorrente compatta (16 unità, dropout, early stopping), con discussione esplicita del rischio di overfitting data la scarsità di dati per un singolo paziente.

**8. Valutazione finale**
Riaddestramento del modello selezionato su train+validation, previsione walk-forward sul test set Kaggle, con analisi separata tra ore realmente osservate e ore interpolate per non falsare la valutazione.

## Risultati

**Gruppo A — Forecast cieco (168h, nessun aggiornamento)**

| Modello | MAE | RMSE | MAPE |
|---|---|---|---|
| **Prophet**  | **2.674** | **3.383** | **39.3%** |
| Prophet + insulina/carboidrati | 2.734 | 3.429 | 40.3% |
| SARIMA | 2.898 | 3.496 | 44.8% |
| SARIMAX + insulina/carboidrati | 2.996 | 3.594 | 46.0% |

**Gruppo B — Walk-forward (1h ahead, aggiornamento orario)**

| Modello | MAE | RMSE | MAPE |
|---|---|---|---|
| **SARIMA**  | **1.666** | **2.342** | **22.7%** |
| LSTM | 1.724 | 2.247 | 25.3% |
| Gradient Boosting | 1.734 | 2.300 | 25.5% |

**Valutazione finale sul test set Kaggle** (SARIMA walk-forward): MAE = 0.422 sull'intero test set; MAE = 0.508 sulle sole ore realmente osservate dal sensore (429 su 2.869 punti totali, il resto è interpolazione dovuta alla bassa densità di campionamento del test set).

## Key Insights

- **Blind vs walk-forward cambia completamente il ranking dei modelli**: Prophet vince nel forecast a lungo termine senza aggiornamenti, ma è SARIMA a vincere nello scenario realistico di monitoraggio continuo — gran parte del vantaggio apparente dei modelli più flessibili si riduce drasticamente quando il confronto è messo sullo stesso piano informativo.
- **Le variabili esogene (insulina, carboidrati) sono statisticamente significative ma non migliorano la previsione fuori campione** — un risultato confermato indipendentemente da due famiglie di modelli (SARIMAX e Prophet+regressori), entrambe peggiorate rispetto alla versione univariata.
- **L'analisi CCF rivela il vero lag causale dell'insulina** (2-3 ore, non 0 come l'intuizione iniziale suggeriva), coerente con la farmacocinetica reale delle insuline rapide — un esempio concreto di come una correlazione a lag 0 possa riflettere causalità inversa (glicemia alta → somministrazione di insulina) più che l'effetto farmacologico vero.
- **Tutti i modelli sottostimano sistematicamente gli eventi estremi** (iper/ipoglicemia severa), il limite clinicamente più rilevante del progetto; una loss pesata e la quantile regression attenuano ma non eliminano il problema.
- **Un caso di data leakage individuato e corretto** tramite feature importance anomala (Gradient Boosting), documentato esplicitamente nel notebook come parte del processo, non nascosto.
- **LSTM non batte un SARIMA ben specificato** con ~2.000 osservazioni — coerente con l'aspettativa teorica che le reti ricorrenti richiedano volumi di dati molto maggiori per esprimere un vantaggio reale.

## Limiti

- Analisi calibrata su un singolo paziente (`p01`); non generalizzabile automaticamente ad altri individui
- Un solo split train/validation (7 giorni); una validazione rolling-origin multi-finestra sarebbe più robusta
- Bias sistematico residuo sugli eventi estremi, solo parzialmente corretto

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `statsmodels` · `pmdarima` · `Prophet` · `LightGBM` · `SHAP` · `TensorFlow/Keras` · `scikit-learn` · `matplotlib` · `seaborn`

## 📂 Struttura del repository

```
├── notebook/
│   └── Diabet_Analysis ITA.ipynb
│   └── Diabet_Analysis ITA.pdf
├── data/
│   └── (dataset BrisT1D — vedi sezione Dataset)
│   
├── requirements.txt
└── README.md
```

## ▶️ Come eseguire il progetto

```bash
git clone https://github.com/elenacascone/<nome-repo>.git
cd <nome-repo>
pip install -r requirements.txt
jupyter notebook notebook/Diabet_Analysis.ipynb
```

## 👤 Autore

**Elena Cascone**
[LinkedIn](https://www.linkedin.com/in/elena-cascone-18ec/) · [GitHub](https://github.com/elenacascone)
