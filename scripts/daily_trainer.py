import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import datetime
from pathlib import Path

# Add project root to python path so we can import src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_preprocessing import list_available_companies, load_stock_data
from src.feature_engineering import create_features
from src.model_training import train_regression_models
from src.explainability import get_global_feature_importance
from src.backtest import Backtester

MODELS_DIR = os.path.join(project_root, "models")
CACHE_DIR = os.path.join(project_root, "cache")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

def train_and_cache_all():
    print(f"Starting daily training run at {datetime.datetime.now()}")
    
    companies = list_available_companies()
    tickers = [c.get('Symbol', '') for c in companies] if companies and isinstance(companies[0], dict) else companies
    
    # For speed in this demo, let's just train the first 5 tickers if it takes too long,
    # but we'll try to do all of them.
    for ticker in tickers:
        print(f"--- Processing {ticker} ---")
        try:
            df = load_stock_data(ticker)
            if df.empty or len(df) < 50:
                print(f"Not enough data for {ticker}, skipping.")
                continue
                
            features = create_features(df).replace([np.inf, -np.inf], np.nan).dropna()
            
            # 1. Train Models
            models = train_regression_models(features, target_col="Close")
            
            # Extract metrics for caching
            metrics_results = {}
            for name, info in models.items():
                metrics_results[name] = {
                    "rmse": float(info.get("rmse", 0)),
                    "mae": float(info.get("mae", 0)),
                    "r2": float(info.get("r2", 0))
                }
                # Save model
                joblib.dump(info['model'], os.path.join(MODELS_DIR, f"{ticker}_{name}.pkl"))
            
            with open(os.path.join(CACHE_DIR, f"{ticker}_metrics.json"), "w") as f:
                json.dump({"status": "success", "models": metrics_results}, f)
            
            rf_info = models.get('Random_Forest', list(models.values())[0])
            model = rf_info['model']
            X_test = rf_info['X_test']
            y_pred = pd.Series(rf_info['y_pred'], index=X_test.index)
            
            # 2. Calculate SHAP & Cache
            importances = get_global_feature_importance(model, X_test)
            top_15 = importances.head(15).to_dict()
            with open(os.path.join(CACHE_DIR, f"{ticker}_shap.json"), "w") as f:
                json.dump({"importances": top_15}, f)
                
            # 3. Calculate Backtest & Cache
            bt_df = features.loc[y_pred.index].copy()
            bt = Backtester(bt_df, y_pred, transaction_cost=0.001)
            results = bt.run()
            
            results['Date'] = results.index.astype(str)
            curve_data = results[['Date', 'Strategy_Equity', 'BuyHold_Equity']].to_dict(orient="records")
            bt_metrics = bt.get_metrics()
            
            with open(os.path.join(CACHE_DIR, f"{ticker}_backtest.json"), "w") as f:
                json.dump({"curve": curve_data, "metrics": bt_metrics}, f)
                
            print(f"Successfully processed {ticker}")
            
        except Exception as e:
            print(f"Failed processing {ticker}: {str(e)}")

if __name__ == "__main__":
    train_and_cache_all()
    print("Daily training complete.")
