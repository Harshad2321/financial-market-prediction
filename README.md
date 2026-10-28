# Market ML: Cloud-Native Market Prediction & Risk Analysis Platform

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI_0.110-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Container-Docker_MultiStage-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)
[![AWS](https://img.shields.io/badge/Cloud-AWS_EC2_%7C_ALB-FF9900.svg?logo=amazon-aws&logoColor=white)](https://aws.amazon.com)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost_%7C_RandomForest-blue.svg)](https://scikit-learn.org)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org)

**Market ML** is an institutional-grade, cloud-native financial market prediction, portfolio optimization, and quantitative risk management platform built for National Stock Exchange (NSE) blue-chip equities.

**Lead Developer & Architect:** [Naman Vyas (@naman00008)](https://github.com/naman00008)

---

## 🚀 Key Architectural Capabilities

### 1. Modern Portfolio Theory (MPT) Optimization & Efficient Frontier
- Vectorized Monte Carlo portfolio simulation across the unit simplex using the **Dirichlet Distribution** ($N = 4,000$ iterations).
- Computes the **Maximum Sharpe Ratio (Tangency Portfolio)** against the benchmark Indian 10-Year Government Securities yield ($R_f = 6.5\%$).
- Implements the **Global Minimum Variance Portfolio** for risk-off capital preservation.
- **Discrete Share Rebalancing**: Translates continuous theoretical weights into executable whole-share order allocations without fractional shares for the Indian stock market.

### 2. Sector Concentration & Black Swan Stress Testing
- **Herfindahl–Hirschman Index (HHI)**: Quantifies economic sector concentration on a normalized $[0, 1]$ scale with a 25% single-industry regulatory alert.
- **Parametric Value at Risk (VaR)** & **Conditional Value at Risk (CVaR / Expected Shortfall)** at a 95% confidence level over 1-day and 1-month horizons.
- **Monte Carlo Simulation**: 1,000 forward-looking price paths generated using Geometric Brownian Motion (GBM) stochastic differential equations.
- **Black Swan Crisis Replay**: Simulates systemic portfolio shocks based on the 2008 Global Financial Crisis ($-55.0\%$), the 2020 COVID-19 Crash ($-38.4\%$), and Crude Oil Spikes ($+\$115/\text{bbl}$).

### 3. Machine Learning & Predictive Modeling
- Ensemble regression modeling utilizing **Random Forest**, **Gradient Boosted Decision Trees**, and **XGBoost**.
- Predictive feature engineering incorporating technical indicators (RSI, Bollinger Bands, MACD, Simple Moving Averages).
- Feature importance analysis and directional accuracy ($DA$) scoring.

### 4. Zero-Knowledge Cryptographic Authentication
- Cryptographic password protection using **PBKDF2-HMAC-SHA256** with **100,000 iterations**.
- Individual 16-byte cryptographically secure random salts per user (`secrets.token_hex(16)`).
- Constant-time verification using `secrets.compare_digest` to neutralize timing side-channel attacks.
- Ephemeral 48-character hexadecimal session tokens (`secrets.token_hex(24)`) and parameterized SQLite persistence (`data/users.db`).

### 5. Cloud-Native & Production Infrastructure
- Asynchronous high-concurrency **FastAPI** web framework running on **Uvicorn**.
- Containerized via a multi-stage **Dockerfile** (`python:3.11-slim`) with automated health checks.
- Deployed on **AWS EC2** behind an **AWS Application Load Balancer (ALB)** with Auto Scaling support.

---

## 📁 Repository Structure

```text
├── app/
│   ├── api_routes.py         # Financial telemetry and company profile REST endpoints
│   ├── auth_routes.py        # Authentication & authorization endpoints (signup, login, me)
│   ├── forecast_routes.py    # Multi-day ML forecasting controllers
│   ├── portfolio_routes.py   # MPT optimization, HHI, VaR, CVaR, and Monte Carlo engine
│   ├── static/               # Client stylesheets (style_v3.css) & UI controllers (script_v52.js)
│   └── templates/            # Single-page application dark theme dashboard (index.html)
├── src/
│   ├── auth.py               # PBKDF2-HMAC-SHA256 password hashing & SQLite database manager
│   ├── tracker.py            # User activity auditing & dossier generator
│   ├── data_preprocessing.py # Market data cleaning, imputation, and log-return calculations
│   ├── feature_engineering.py# Technical momentum indicators (RSI, MACD, Bollinger Bands)
│   ├── model_training.py     # Supervised ML training pipelines (RF, GBR, XGBoost)
│   ├── backtest.py           # Algorithmic strategy backtester (Sortino, Sharpe, Max Drawdown)
│   ├── explainability.py     # Global feature importance scoring
│   └── live_data.py          # Real-time yfinance telemetry & VADER sentiment analysis
├── tests/
│   ├── test_auth_system.py   # Security and password hashing unit test suite
│   ├── test_market_api_routes.py
│   └── test_portfolio_optimizer_routes.py
├── showcase_all.py           # Master 8-phase interactive demonstration script
├── showcase.sh               # Execution wrapper for demonstration script
├── Dockerfile                # Multi-stage production container build
├── docker-compose.yml        # Docker service definition with port 80/8000 bindings
├── main.py                   # FastAPI ASGI entrypoint and CORS middleware
└── requirements.txt          # Python production dependencies
```

---

## ⚡ Quickstart

### Local Setup
```bash
# Clone the repository
git clone https://github.com/naman00008/financial-market-prediction.git
cd financial-market-prediction

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```
Open [http://localhost:8000](http://localhost:8000) in your web browser.

### Docker Run
```bash
docker compose up -d --build
```
Access the application at [http://localhost:80](http://localhost:80) or [http://localhost:8000](http://localhost:8000).

---

## 🧪 Test Suite
Run the automated test suite to verify the authentication engine and quantitative calculations:
```bash
python3 tests/test_auth_system.py
```
Run the full 8-phase terminal demonstration:
```bash
python3 showcase_all.py
```

---

## 👤 Author
**Naman Vyas**  
GitHub: [@naman00008](https://github.com/naman00008)
