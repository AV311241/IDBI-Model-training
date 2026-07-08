How to Run
# Option 1 — Direct Python
python main.py

# Option 2 — Uvicorn CLI
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Option 3 — Production (4 workers)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
🧪 Test the API — Sample cURL Requests
Single Prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "annual_income"           : 45000,
    "emp_length"              : 3,
    "application_type"        : "individual",
    "loan_amount"             : 15000,
    "term"                    : 36,
    "int_rate"                : 14.5,
    "installment"             : 350.0,
    "purpose"                 : "car",
    "grade"                   : "c",
    "sub_grade"               : "c4",
    "home_ownership"          : "rent",
    "verification_status"     : "source verified",
    "dti"                     : 18.5,
    "total_acc"               : 8,
    "total_payment"           : 5000,
    "address_state"           : "CA",
    "loan_age_months"         : 12,
    "issue_month"             : 3,
    "issue_year"              : 2021,
    "days_since_last_payment" : 90,
    "days_since_credit_pull"  : 30,
    "days_to_next_payment"    : 15
  }'
Health Check
curl http://localhost:8000/health
Model Info
curl http://localhost:8000/model/info
📊 Sample API Response
{
  "prediction"          : 0,
  "decision"            : "APPROVE",
  "risk_band"           : "GREEN",
  "default_probability" : 0.2134,
  "confidence"          : 78.66,
  "health_card"         : {
    "liquidity"   : 72,
    "solvency"    : 65,
    "growth"      : 45,
    "compliance"  : 78,
    "repayment"   : 80,
    "overall"     : 714
  },
  "explanation"         : {
    "top_risk_factors"   : {
      "dti"              :  0.1823,
      "int_rate"         :  0.1245,
      "loan_amount"      :  0.0987,
      "days_since_last_payment": 0.0876,
      "emi_to_income_ratio"    : 0.0654
    },
    "top_safety_factors" : {
      "grade"            : -0.2341,
      "annual_income"    : -0.1876,
      "payment_ratio"    : -0.1543,
      "verification_status": -0.0987,
      "total_acc"        : -0.0654
    },
    "base_value"         : 0.1823
  },
  "recommended_amount"  : 15000,
  "recommended_rate"    : 14.5,
  "recommended_term"    : 36,
  "model_name"          : "XGBoost",
  "model_auc"           : 92.4,
  "request_id"          : "f3a2b1c4-d5e6-7890-abcd-ef1234567890",
  "timestamp"           : "2026-07-08T13:37:00Z"
}
🗺️ Complete Project Flow
loan_data.xlsx
      ↓
data_cleaning.py  →  X_train.csv / X_test.csv / scaler.pkl
      ↓
model_training.py →  best_model.pkl / model_metadata.pkl
      ↓
main.py (FastAPI)
      ↓
┌─────────────────────────────────────────────┐
│  POST /predict        → Single prediction   │
│  POST /predict/batch  → Batch predictions   │
│  POST /health-card    → Health Card only    │
│  POST /explain        → SHAP only           │
│  POST /predict/summary→ Portfolio stats     │
│  GET  /health         → API status          │
│  GET  /model/info     → Model metadata      │
└─────────────────────────────────────────────┘
      ↓
Bank Loan Officer Dashboard / Mobile App
📁 Final Complete File Structure
msme_loan_api/
│
├── main.py                     ← FastAPI app
├── requirements.txt            ← Dependencies
├── README.md                   ← Documentation
│
├── model/
│   ├── best_model.pkl          ← Trained XGBoost model
│   ├── scaler.pkl              ← StandardScaler
│   └── model_metadata.pkl      ← Features & metrics
│
├── schemas/
│   └── loan_schema.py          ← Pydantic request/response models
│
├── services/
│   └── prediction.py           ← Encoding & SHAP logic
│
└── utils/
    └── health_card.py          ← 5D scoring & loan offer logic```
🔒 Production Hardening Checklist
# ============================================================
# production_config.py
# Settings to apply before going live
# ============================================================

# 1. ── Environment Variables ─────────────────────────────────
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME        : str   = "MSME Loan Risk API"
    API_VERSION     : str   = "1.0.0"
    DEBUG           : bool  = False
    API_KEY         : str   = os.getenv("API_KEY", "change-me-in-production")
    MODEL_DIR       : str   = os.getenv("MODEL_DIR", "model/")
    MAX_BATCH_SIZE  : int   = 100
    ALLOWED_ORIGINS : list  = ["https://yourbankdomain.com"]
    LOG_LEVEL       : str   = "warning"

    class Config:
        env_file = ".env"

settings = Settings()
# 2. ── API Key Authentication Middleware ─────────────────────
from fastapi          import Security, HTTPException, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = "Invalid API Key. Access denied."
        )
    return api_key

# Add to any route like:
# @app.post("/predict", dependencies=[Depends(verify_api_key)])
# 3. ── Rate Limiting ─────────────────────────────────────────
# pip install slowapi
from slowapi            import Limiter, _rate_limit_exceeded_handler
from slowapi.util       import get_remote_address
from slowapi.errors     import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add to routes like:
# @app.post("/predict")
# @limiter.limit("60/minute")
# async def predict_loan(request: Request, application: LoanApplicationRequest):
# 4. ── Structured Logging ────────────────────────────────────
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp" : self.formatTime(record),
            "level"     : record.levelname,
            "message"   : record.getMessage(),
            "module"    : record.module,
            "funcName"  : record.funcName,
        }
        return json.dumps(log_record)

handler   = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger    = logging.getLogger("msme_api")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage:
# logger.info({"event": "prediction", "request_id": rid, "decision": decision})
# 5. ── Docker Setup ──────────────────────────────────────────
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run with gunicorn + uvicorn workers for production
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "warning"]
# docker-compose.yml
version: "3.9"

services:
  msme_api:
    build       : .
    container_name: msme_loan_api
    ports       :
      - "8000:8000"
    volumes     :
      - ./model:/app/model       # Mount model files
    environment :
      - API_KEY=your-secret-key
      - MODEL_DIR=model/
      - DEBUG=false
    restart     : unless-stopped
    healthcheck :
      test        : ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval    : 30s
      timeout     : 10s
      retries     : 3
🧪 Python Test Script
# test_api.py
# Run this to test all endpoints after starting the server
# pip install requests

import requests
import json

BASE_URL = "http://localhost:8000"

SAMPLE_APPLICATION = {
    "annual_income"           : 45000,
    "emp_length"              : 3,
    "application_type"        : "individual",
    "loan_amount"             : 15000,
    "term"                    : 36,
    "int_rate"                : 14.5,
    "installment"             : 350.0,
    "purpose"                 : "car",
    "grade"                   : "c",
    "sub_grade"               : "c4",
    "home_ownership"          : "rent",
    "verification_status"     : "source verified",
    "dti"                     : 18.5,
    "total_acc"               : 8,
    "total_payment"           : 5000,
    "address_state"           : "CA",
    "loan_age_months"         : 12,
    "issue_month"             : 3,
    "issue_year"              : 2021,
    "days_since_last_payment" : 90,
    "days_since_credit_pull"  : 30,
    "days_to_next_payment"    : 15
}

def test_health():
    print("\n" + "="*50)
    print("TEST 1: Health Check")
    print("="*50)
    r = requests.get(f"{BASE_URL}/health")
    print(f"Status  : {r.status_code}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    print("✅ PASSED")

def test_model_info():
    print("\n" + "="*50)
    print("TEST 2: Model Info")
    print("="*50)
    r = requests.get(f"{BASE_URL}/model/info")
    print(f"Status  : {r.status_code}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")
    assert r.status_code == 200
    print("✅ PASSED")

def test_single_prediction():
    print("\n" + "="*50)
    print("TEST 3: Single Prediction")
    print("="*50)
    r = requests.post(
        f"{BASE_URL}/predict",
        json    = SAMPLE_APPLICATION,
        headers = {"Content-Type": "application/json"}
    )
    print(f"Status  : {r.status_code}")
    data = r.json()
    print(f"Decision           : {data['decision']}")
    print(f"Risk Band          : {data['risk_band']}")
    print(f"Default Probability: {data['default_probability']}")
    print(f"Health Score       : {data['health_card']['overall']}/1000")
    print(f"Top Risk Factor    : {list(data['explanation']['top_risk_factors'].keys())[0]}")
    assert r.status_code == 200
    assert data["decision"] in ["APPROVE", "MANUAL REVIEW", "REJECT"]
    print("✅ PASSED")

def test_batch_prediction():
    print("\n" + "="*50)
    print("TEST 4: Batch Prediction (```python
def test_batch_prediction():
    print("\n" + "="*50)
    print("TEST 4: Batch Prediction (3 applications)")
    print("="*50)

    # Create 3 slightly different applications
    app1 = SAMPLE_APPLICATION.copy()
    app2 = {**SAMPLE_APPLICATION, "annual_income": 80000, "dti": 8.5,  "grade": "a", "sub_grade": "a1"}
    app3 = {**SAMPLE_APPLICATION, "annual_income": 20000, "dti": 45.0, "grade": "g", "sub_grade": "g5"}

    r = requests.post(
        f"{BASE_URL}/predict/batch",
        json    = {"applications": [app1, app2, app3]},
        headers = {"Content-Type": "application/json"}
    )
    print(f"Status             : {r.status_code}")
    data = r.json()
    print(f"Total Applications : {data['total_applications']}")
    print(f"Approved           : {data['approved']}")
    print(f"Manual Review      : {data['manual_review']}")
    print(f"Rejected           : {data['rejected']}")
    print(f"Processing Time    : {data['processing_time_ms']} ms")
    for res in data["results"]:
        print(f"  → App {res['index']}: {res['decision']:<15} | Risk: {res['risk_band']:<6} | Prob: {res['default_probability']:.2%} | Health: {res['overall_health_score']}/1000")
    assert r.status_code == 200
    assert data["total_applications"] == 3
    print("✅ PASSED")

def test_health_card():
    print("\n" + "="*50)
    print("TEST 5: Financial Health Card")
    print("="*50)
    r = requests.post(
        f"{BASE_URL}/health-card",
        json    = SAMPLE_APPLICATION,
        headers = {"Content-Type": "application/json"}
    )
    print(f"Status  : {r.status_code}")
    data = r.json()
    print(f"Liquidity   : {data['liquidity']}/100")
    print(f"Solvency    : {data['solvency']}/100")
    print(f"Growth      : {data['growth']}/100")
    print(f"Compliance  : {data['compliance']}/100")
    print(f"Repayment   : {data['repayment']}/100")
    print(f"Overall     : {data['overall']}/1000")
    assert r.status_code == 200
    assert 0 <= data["overall"] <= 1000
    print("✅ PASSED")

def test_explain():
    print("\n" + "="*50)
    print("TEST 6: SHAP Explanation")
    print("="*50)
    r = requests.post(
        f"{BASE_URL}/explain",
        json    = SAMPLE_APPLICATION,
        headers = {"Content-Type": "application/json"}
    )
    print(f"Status  : {r.status_code}")
    data = r.json()
    print(f"Base Value         : {data['base_value']}")
    print(f"Top Risk Factors   :")
    for feat, val in data["top_risk_factors"].items():
        print(f"  → {feat:<35} : +{val:.4f}")
    print(f"Top Safety Factors :")
    for feat, val in data["top_safety_factors"].items():
        print(f"  → {feat:<35} :  {val:.4f}")
    assert r.status_code == 200
    print("✅ PASSED")

def test_portfolio_summary():
    print("\n" + "="*50)
    print("TEST 7: Portfolio Summary")
    print("="*50)

    apps = []
    for i in range(10):
        app = SAMPLE_APPLICATION.copy()
        app["annual_income"] = 20000 + (i * 5000)
        app["dti"]           = 5.0   + (i * 3.5)
        apps.append(app)

    r = requests.post(
        f"{BASE_URL}/predict/summary",
        json    = {"applications": apps},
        headers = {"Content-Type": "application/json"}
    )
    print(f"Status  : {r.status_code}")
    data = r.json()
    print(f"Total Applications     : {data['total_applications']}")
    print(f"Approval Rate          : {data['portfolio_summary']['approval_rate_pct']}%")
    print(f"Rejection Rate         : {data['portfolio_summary']['rejection_rate_pct']}%")
    print(f"Avg Default Probability: {data['risk_statistics']['avg_default_probability']:.2%}")
    print(f"Avg Health Score       : {data['health_card_statistics']['avg_health_score']}/1000")
    print(f"Risk Band Distribution :")
    print(f"  🟢 GREEN : {data['risk_band_distribution']['GREEN']}")
    print(f"  🟡 AMBER : {data['risk_band_distribution']['AMBER']}")
    print(f"  🔴 RED   : {data['risk_band_distribution']['RED']}")
    assert r.status_code == 200
    print("✅ PASSED")

def test_invalid_input():
    print("\n" + "="*50)
    print("TEST 8: Invalid Input Handling")
    print("="*50)

    bad_app = SAMPLE_APPLICATION.copy()
    bad_app["annual_income"] = -5000   # Invalid: negative income
    bad_app["grade"]         = "z"     # Invalid: unknown grade

    r = requests.post(
        f"{BASE_URL}/predict",
        json    = bad_app,
        headers = {"Content-Type": "application/json"}
    )
    print(f"Status  : {r.status_code}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")
    assert r.status_code == 422   # Unprocessable Entity
    print("✅ PASSED — Invalid input correctly rejected")


# ── Run All Tests ────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "🧪 " * 20)
    print("MSME LOAN RISK API — FULL TEST SUITE")
    print("🧪 " * 20)

    tests = [
        test_health,
        test_model_info,
        test_single_prediction,
        test_batch_prediction,
        test_health_card,
        test_explain,
        test_portfolio_summary,
        test_invalid_input,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR : {e}")
            failed += 1

    print("\n" + "="*50)
    print(f"📋 TEST RESULTS: {passed} passed | {failed} failed")
    print("="*50)
    if failed == 0:
        print("🎉 All tests passed! API is ready for deployment.")
    else:
        print("⚠️  Some tests failed. Check logs above.")
🗺️ Complete End-to-End Summary
┌─────────────────────────────────────────────────────────────┐
│              MSME LOAN RISK ASSESSMENT SYSTEM               │
│                   Complete Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📊 DATA PIPELINE                                           │
│  loan_data.xlsx (38K)                                       │
│       ```
│       ↓                                                     │
│  data_cleaning.py                                           │
│  ├─ Remove duplicates & IDs                                 │
│  ├─ Clean emp_length, term, int_rate, dti                   │
│  ├─ Parse DD-MM-YYYY dates                                  │
│  ├─ Engineer 9 new features                                 │
│  ├─ Encode 9 categorical columns                            │
│  ├─ Impute nulls (median/mode)                              │
│  ├─ Cap outliers (IQR)                                      │
│  ├─ SMOTE for class balance                                 │
│  └─ StandardScaler                                          │
│       ↓                                                     │
│  X_train.csv / X_test.csv / y_train.csv / y_test.csv        │
│  scaler.pkl                                                 │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🤖 MODEL PIPELINE                                          │
│  model_training.py                                          │
│  ├─ Train 4 models (LR, RF, XGBoost, LightGBM)             │
│  ├─ Compare AUC / F1 / Accuracy / Precision / Recall        │
│  ├─ Select best model (XGBoost ~92% AUC)                    │
│  ├─ SHAP explainability                                     │
│  ├─ Financial Health Card (5D radar)                        │
│  └─ Save best_model.pkl + model_metadata.pkl                │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🚀 API LAYER (FastAPI)                                     │
│  main.py                                                    │
│  ├─ POST /predict         → Single decision                 │
│  ├─ POST /predict/batch   → Up to 100 applications          │
│  ├─ POST /predict/summary → Portfolio risk stats            │
│  ├─ POST /health-card     → 5D health score only            │
│  ├─ POST /explain         → SHAP explanation only           │
│  ├─ GET  /health          → API status                      │
│  └─ GET  /model/info      → Model metadata                  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🔒 PRODUCTION FEATURES                                     │
│  ├─ API Key Authentication                                  │
│  ├─ Rate Limiting (60 req/min)                              │
│  ├─ CORS Middleware                                         │
│  ├─ GZip Compression                                        │
│  ├─ Request Logging                                         │
│  ├─ Global Exception Handler                                │
│  ├─ Docker + docker-compose                                 │
│  └─ Structured JSON Logging                                 │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📋 RESPONSE INCLUDES                                       │
│  ├─ Decision      : APPROVE / MANUAL REVIEW / REJECT        │
│  ├─ Risk Band     : 🟢 GREEN / 🟡 AMBER / 🔴 RED            │
│  ├─ Default Prob  : 0.0 – 1.0                               │
│  ├─ Health Card   : Liquidity, Solvency, Growth,            │
│  │                  Compliance, Repayment (0–1000)          │
│  ├─ SHAP          : Top 5 risk & safety factors             │
│  ├─ Loan Offer    : Amount, Rate, Term (if approved)        │
│  └─ Audit Trail   : request_id + timestamp                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
⚡ Quick Commands Reference
# ── Setup ────────────────────────────────────────────────────
pip install -r requirements.txt

# ── Run API (Development) ────────────────────────────────────
python main.py

# ── Run API (Production) ─────────────────────────────────────
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# ── Run with Docker ──────────────────────────────────────────
docker build -t msme-loan-api .
docker run -p 8000:8000 -v $(pwd)/model:/app/model msme-loan-api

# ── Run with Docker Compose ──────────────────────────────────
docker-compose up --build

# ── Run Tests ────────────────────────────────────────────────
python test_api.py

# ── View API Docs ────────────────────────────────────────────
open http://localhost:8000/docs       # Swagger UI
open http://localhost:8000/redoc      # ReDoc UI
Abhishek, your complete system is now ready across 3 scripts + 1 API:

Search
Script
Purpose
data_cleaning.py	Clean & encode 38K loan records
model_training.py	Train XGBoost + SHAP + Health Card
main.py (FastAPI)	Serve predictions via REST API
test_api.py	Validate all 7 endpoints