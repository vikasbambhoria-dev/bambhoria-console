"""
neural_insight_engine.py
Bambhoria Neural Insight Engine v1.0
Author: Vikas Bambhoria
Purpose:
 - Analyse historical trades & signals
 - Learn winning patterns using AI
 - Suggest strategy optimizations
"""

import pandas as pd, numpy as np, json, os, time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
from pathlib import Path

ORDERS_LOG = Path("logs/orders_log.json")
INSIGHT_MODEL = Path("models/insight_pattern_model.pkl")
INSIGHT_REPORT = Path("reports/insight_report.txt")
os.makedirs(INSIGHT_REPORT.parent, exist_ok=True)
os.makedirs(INSIGHT_MODEL.parent, exist_ok=True)

# --------- Helper: Load & prepare trade data ---------
def load_trades():
    if not ORDERS_LOG.exists(): 
        print("❌ No order log found."); 
        return pd.DataFrame()
    data = [json.loads(x) for x in open(ORDERS_LOG,"r") if x.strip()]
    df = pd.DataFrame(data)
    if df.empty: 
        return df
    df["profit_flag"] = (df["pnl"]>0).astype(int)
    df["hour"] = pd.to_datetime(df["time"]).dt.hour
    df["side_num"] = df["side"].map({"BUY":1,"SELL":-1})
    df["symbol_code"] = df["symbol"].astype("category").cat.codes
    return df

# --------- Feature builder ---------
def build_features(df):
    return df[["symbol_code","side_num","hour","qty","price"]], df["profit_flag"]

# --------- Train & evaluate pattern model ---------
def train_pattern_model():
    df = load_trades()
    if len(df)<10:
        print("⚠️ Not enough trades for training."); 
        return
    X,y = build_features(df)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X,y)
    preds = model.predict(X)
    acc = accuracy_score(y,preds)
    rep = classification_report(y,preds,zero_division=0)
    dump(model, INSIGHT_MODEL)
    with open(INSIGHT_REPORT,"w") as f:
        f.write(f"Training time: {datetime.now()}\n")
        f.write(f"Accuracy: {acc:.3f}\n")
        f.write(rep)
    print(f"✅ Insight model trained. Accuracy={acc:.2f}")
    print(rep)

# --------- Insight inference ---------
def suggest_insight(symbol:str, side:str, hour:int, qty:int, price:float):
    if not INSIGHT_MODEL.exists(): 
        print("⚠️ No trained model, training now..."); 
        train_pattern_model()
    model = load(INSIGHT_MODEL)
    df = pd.DataFrame([{
        "symbol_code": hash(symbol)%1000,
        "side_num": 1 if side=="BUY" else -1,
        "hour": hour,
        "qty": qty,
        "price": price
    }])
    prob = model.predict_proba(df)[0][1]
    suggestion = "✅ Favorable Setup" if prob>0.6 else "⚠️ Risky Trade"
    print(f"{symbol} {side} @ {price} → {suggestion} (Success Prob {prob:.2f})")
    return {"symbol":symbol,"side":side,"prob":round(prob,2),"suggestion":suggestion}

# --------- Batch analysis ---------
def analyse_trends():
    df = load_trades()
    if df.empty: 
        print("No trades for analysis."); return
    stats = df.groupby("symbol")["pnl"].agg(["count","sum","mean"]).sort_values("mean",ascending=False)
    print("📊 Symbol Performance:")
    print(stats.head(10))
    df["hour"] = pd.to_datetime(df["time"]).dt.hour
    hour_pnl = df.groupby("hour")["pnl"].mean()
    print("⏰ Best Trading Hours:")
    print(hour_pnl.sort_values(ascending=False).head(5))

# --------- Main ---------
if __name__=="__main__":
    print("🧬 Bambhoria Neural Insight Engine v1.0 starting...")
    train_pattern_model()
    analyse_trends()
    suggest_insight("RELIANCE","BUY",10,10,2550)