"""
signal_generator.py
Bambhoria AI Signal Generator v1.0
Author: Vikas Bambhoria
Purpose:
 - Load trained ML model (LightGBM)
 - Predict BUY / SELL / HOLD on incoming tick stream
 - Auto-execute orders via OrderBrain (paper / live)
"""

import joblib, os, time, numpy as np, pandas as pd
from core_engine.order_control import OrderBrain
from core_engine.risk_manager import RiskManager

# ---------- CONFIG ----------
MODEL_PATH = "models/godeye_lgbm_model.pkl"
FEATURE_ORDER = ["price_change","volume_change","volatility"]
EXECUTION_MODE = "paper"      # change to "live" when API ready
MIN_CONFIDENCE = 0.65         # threshold for trade trigger
COOLDOWN_SECONDS = 10         # min gap between same-symbol trades

brain = OrderBrain(mode=EXECUTION_MODE)
risk  = RiskManager()
last_trade_time = {}

# ---------- LOAD MODEL ----------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print(f"✅ Model loaded: {MODEL_PATH}")

# ---------- FEATURE BUILDER ----------
def make_features(tick):
    lp   = tick["last_price"]
    ohlc = tick.get("ohlc",{})
    vol  = tick.get("volume",0)
    base = ohlc.get("open",lp)
    return pd.DataFrame([{
        "price_change": (lp-base)/base,
        "volume_change": vol/10000,
        "volatility": abs(ohlc.get("high",lp)-ohlc.get("low",lp))/base
    }], columns=FEATURE_ORDER)

# ---------- SIGNAL GENERATION ----------
def generate_signal(tick):
    X = make_features(tick)
    prob = model.predict_proba(X)[0] if hasattr(model,"predict_proba") else [0,0,0]
    signal_idx = int(np.argmax(prob))
    conf = float(np.max(prob))
    labels = {0:"SELL",1:"HOLD",2:"BUY"} if len(prob)==3 else {0:"SELL",1:"BUY"}
    signal = labels.get(signal_idx,"HOLD")
    return signal, conf

# ---------- EXECUTION ----------
def process_tick(tick):
    sym = tick["symbol"]
    signal, conf = generate_signal(tick)
    now = time.time()
    if conf < MIN_CONFIDENCE or signal == "HOLD":
        return
    # cooldown check
    if sym in last_trade_time and now - last_trade_time[sym] < COOLDOWN_SECONDS:
        return
    last_trade_time[sym] = now

    side = signal
    qty  = 10
    price = tick["last_price"]
    print(f"📈 Signal: {sym} {signal} (conf={conf:.2f}) @ {price}")
    
    # Send signal to dashboard
    try:
        import requests
        signal_data = {
            "symbol": sym,
            "signal": signal,
            "confidence": conf,
            "price": price,
            "quantity": qty,
            "timestamp": time.time()
        }
        requests.post("http://localhost:5006/api/signals", json=signal_data, timeout=1)
    except:
        pass  # Dashboard posting is optional
    
    brain.place_order(sym, side, qty, price)

# ---------- STREAM INTERFACE ----------
def run_live_simulator(feed_source):
    """
    feed_source : iterable or generator yielding tick dicts
    e.g. from mock_live_feed or backtest_replay_engine
    """
    for tick in feed_source:
        process_tick(tick)
        time.sleep(0.5)

# ---------- DEMO ----------
if __name__ == "__main__":
    # quick demo with fake ticks
    import random
    fake_ticks = [{
        "symbol":"RELIANCE",
        "last_price":2550+random.uniform(-2,2),
        "volume":12000,
        "ohlc":{"open":2550,"high":2555,"low":2548}
    } for _ in range(20)]
    run_live_simulator(fake_ticks)