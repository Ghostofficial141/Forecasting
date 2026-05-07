# Sales Forecasting System

> **Production-ready end-to-end time-series forecasting platform** using SARIMA, Facebook Prophet, XGBoost, and LSTM — with a FastAPI REST service, full MLOps pipeline, and Docker deployment.

---

## 📐 Architecture

```
Raw Data (Excel)
     │
     ▼
┌─────────────────────────────────────────────────┐
│              TRAINING PIPELINE                  │
│                                                 │
│  DataIngestion  →  DataValidation               │
│       ↓                                         │
│  Preprocessing  →  FeatureEngineering           │
│       ↓                                         │
│  ModelTraining (SARIMA + Prophet + XGB + LSTM)  │
│       ↓                                         │
│  ModelEvaluation  →  ModelSelection             │
│       ↓                                         │
│  Artifacts (models/, metrics/, forecasts/)      │
└─────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────┐
│            PREDICTION PIPELINE / API            │
│                                                 │
│  FastAPI REST Service                           │
│  POST /predict  →  PredictionPipeline           │
│                    → ModelSelector              │
│                    → Best Model Inference       │
│                    → Forecast JSON              │
└─────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
sales_forecasting_system/
├── data/
│   ├── raw/                      ← Place your Excel file here
│   ├── processed/                ← Auto-generated Parquet snapshots
│   └── external/                 ← Holiday calendars, etc.
├── notebooks/                    ← Exploratory analysis notebooks
├── src/
│   ├── components/
│   │   ├── data_ingestion.py     ← Load + auto-detect columns
│   │   ├── data_validation.py    ← Schema & quality checks
│   │   ├── preprocessing.py      ← Gap fill, impute, scale
│   │   ├── feature_engineering.py← Lag, rolling, calendar features
│   │   ├── model_training.py     ← Train all 4 models per state
│   │   ├── model_evaluation.py   ← RMSE/MAE/MAPE/R2 per model/state
│   │   ├── model_selection.py    ← Pick best model per state
│   │   └── prediction.py         ← Unified inference interface
│   ├── models/
│   │   ├── arima_model.py        ← SARIMA / auto_arima
│   │   ├── prophet_model.py      ← Facebook Prophet
│   │   ├── xgboost_model.py      ← XGBoost with recursive forecasting
│   │   └── lstm_model.py         ← Multi-layer LSTM (Keras/TF)
│   ├── pipelines/
│   │   ├── training_pipeline.py  ← 7-step training orchestrator
│   │   └── prediction_pipeline.py← Inference orchestrator
│   ├── utils/
│   │   ├── logger.py             ← Structured rotating file logger
│   │   ├── exception.py          ← Custom exception hierarchy
│   │   ├── helpers.py            ← File I/O, column detection, splits
│   │   └── metrics.py            ← RMSE, MAE, MAPE, SMAPE, R2
│   ├── config/config.yaml        ← All tuneable parameters
│   └── constants/__init__.py     ← Project-wide paths and labels
├── api/
│   ├── app.py                    ← FastAPI factory + middleware
│   ├── routes/forecast.py        ← All REST endpoints
│   └── schemas/
│       ├── request.py            ← Pydantic request models
│       └── response.py           ← Pydantic response models
├── artifacts/
│   ├── models/                   ← Trained model files (.pkl / .keras)
│   ├── metrics/                  ← Evaluation JSON/CSV
│   └── forecasts/                ← Forecast outputs
├── tests/
│   └── test_components.py        ← pytest unit tests
├── logs/                         ← Rotating log files
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.py
├── setup.cfg
├── .gitignore
├── main.py                       ← CLI entry point
└── README.md
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9–3.11
- pip
- (Optional) Docker + Docker Compose

### 1. Clone / enter project directory
```bash
cd sales_forecasting_system
```

### 2. Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 4. Place your data
Copy the Excel file into `data/raw/`:
```
data/raw/Forecasting Case- Study (1).xlsx
```

> **Note:** The system auto-detects the date, sales, and state columns. You can override these in `src/config/config.yaml`.

### 5. Train all models
```bash
python main.py train
```
This runs all 7 pipeline steps and saves artifacts to `artifacts/`.

### 6. Start the REST API
```bash
python main.py api
# or
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

**Open Swagger UI:** http://localhost:8000/docs

---

## 🔧 Configuration (`src/config/config.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `data.raw_data_path` | `data/raw/...xlsx` | Path to raw Excel file |
| `data.frequency` | `W` | Time series frequency |
| `data.test_ratio` | `0.20` | Validation split fraction |
| `models.forecast_horizon` | `8` | Weeks to forecast |
| `models.arima.enabled` | `true` | Enable SARIMA |
| `models.prophet.enabled` | `true` | Enable Prophet |
| `models.xgboost.enabled` | `true` | Enable XGBoost |
| `models.lstm.enabled` | `true` | Enable LSTM |
| `selection.primary_metric` | `RMSE` | Metric for best model selection |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check + model readiness |
| `POST` | `/predict` | Forecast for a state |
| `POST` | `/predict/all` | Forecast for ALL states |
| `POST` | `/train` | Retrain all models |
| `GET` | `/metrics` | All model evaluation metrics |
| `GET` | `/metrics/best` | Best model per state |
| `GET` | `/models` | Available model descriptions |
| `GET` | `/docs` | Swagger interactive UI |
| `GET` | `/redoc` | ReDoc documentation |

### Example: POST /predict
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"state": "California", "weeks": 8}'
```

**Response:**
```json
{
  "status": "success",
  "state": "California",
  "model_used": "Prophet",
  "weeks": 8,
  "forecast": [
    {"date": "2024-02-05", "sales": 12345.67},
    {"date": "2024-02-12", "sales": 12456.78},
    ...
  ],
  "generated_at": "2024-01-29T10:00:00"
}
```

### Example: GET /metrics
```bash
curl http://localhost:8000/metrics
```

### Example: Force specific model
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"state": "Texas", "weeks": 4, "model": "XGBoost"}'
```

---

## 🤖 Model Details

| Model | Type | Key Features |
|-------|------|-------------|
| **SARIMA** | Statistical | Auto-order selection (pmdarima), seasonal (m=52) |
| **Prophet** | Statistical/ML | Yearly + weekly seasonality, US holidays, CI |
| **XGBoost** | ML | Lag + rolling + calendar features, recursive forecasting |
| **LSTM** | Deep Learning | Multi-layer + dropout, sequence-to-one, MinMax scaling |

### Feature Engineering
- **Lag features:** `lag_1`, `lag_7`, `lag_30`
- **Rolling stats:** `rolling_mean_7/30`, `rolling_std_7/30` (shifted 1 step to prevent leakage)
- **Calendar:** `day_of_week`, `week_of_year`, `month`, `quarter`, `year`, `is_weekend`
- **Holidays:** `holiday_flag` (US Federal Holidays via `holidays` library)

### No Data Leakage
- Strict chronological train/validation split (no shuffle)
- Rolling features shifted by 1 step
- Scalers fitted only on training data

---

## 🐳 Docker Deployment

### Build and run the API
```bash
docker compose up --build forecasting-api
```

### Train models in Docker
```bash
docker compose --profile train up forecasting-trainer
```

### Check API health
```bash
curl http://localhost:8000/
```

---

## 🧪 Running Tests
```bash
pytest tests/ -v --cov=src
```

Expected output:
```
tests/test_components.py::TestHelpers::test_detect_date_column PASSED
tests/test_components.py::TestMetrics::test_rmse_perfect PASSED
...
Coverage: 85%+
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error — primary selection metric |
| **MAE** | Mean Absolute Error |
| **MAPE** | Mean Absolute Percentage Error |
| **SMAPE** | Symmetric MAPE (bounded 0–200%) |
| **R²** | Coefficient of determination |

Results saved in:
- `artifacts/metrics/all_model_metrics.json` / `.csv`
- `artifacts/metrics/best_model_selection.json`

---

## 🐛 Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| `FileNotFoundError: Processed data not found` | Run `python main.py train` first |
| `Model selection file not found` | Run `python main.py train` first |
| `pmdarima not found` | `pip install pmdarima` |
| `prophet not found` | `pip install prophet` |
| `tensorflow not found` | `pip install tensorflow` |
| `No module named 'src'` | Run from project root or `pip install -e .` |
| LSTM training very slow | Reduce `epochs` in config.yaml |
| SARIMA auto_arima hangs | Set `max_p: 2, max_q: 2` in config.yaml |

---

## 🔮 Assumptions

1. Data contains at minimum: date column, numeric sales column, and optionally a state/region column.
2. Weekly frequency data is expected; the system can adapt to daily/monthly via config.
3. Missing values are filled by linear interpolation then forward/backward fill.
4. Outliers are winsorised at 1st/99th percentile per state.
5. LSTM requires `sequence_length` ≤ training data length (default 12 weeks minimum).

---

## 🔧 Future Improvements

- [ ] Hyperparameter tuning with Optuna / Ray Tune
- [ ] MLflow experiment tracking
- [ ] Real-time streaming predictions (Kafka integration)
- [ ] Ensemble forecasting (weighted average of all models)
- [ ] Auto-retraining scheduler (APScheduler / Celery)
- [ ] Grafana dashboard for live monitoring
- [ ] CI/CD with GitHub Actions
- [ ] Model drift detection
- [ ] Feature store (Feast / Tecton)

---

## 📋 CLI Reference

```bash
python main.py train      # Full training pipeline
python main.py predict    # Generate forecasts for all states
python main.py api        # Start REST API server
python main.py all        # Train then predict
```

---

*Built as a FAANG-level production forecasting system — modular, scalable, observable.*
