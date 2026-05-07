# ── Build args ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="ml-team@company.com"
LABEL description="Sales Forecasting API — SARIMA, Prophet, XGBoost, LSTM"
LABEL version="1.0.0"

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    libffi-dev \
    libssl-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps (cached layer) ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
# Install in two phases to leverage Docker cache
RUN pip install --no-cache-dir \
    numpy pandas scikit-learn scipy pyarrow openpyxl \
    pyyaml python-dotenv joblib tqdm rich holidays
RUN pip install --no-cache-dir \
    statsmodels pmdarima prophet xgboost
RUN pip install --no-cache-dir \
    tensorflow
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" pydantic python-multipart httpx

# ── Copy project ─────────────────────────────────────────────────────────────
COPY . .

# ── Editable install ──────────────────────────────────────────────────────────
RUN pip install -e . --no-deps

# ── Create required directories ───────────────────────────────────────────────
RUN mkdir -p logs artifacts/models artifacts/metrics artifacts/forecasts \
    artifacts/plots data/raw data/processed data/external

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
