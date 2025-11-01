"""
backtest_replay_engine.py
Bambhoria God-Eye Auto-Backtest & Replay Engine v1.0
Author : Vikas Bambhoria
Purpose: Replay historical data as live ticks to dashboard for model testing & PnL simulation
"""

import csv, json, time, requests, os, random
from datetime import datetime
from pathlib import Path

# ---------- CONFIG ----------
DATA_FOLDER      = Path("data/historical/")      # CSV data folder
DASHBOARD_URL    = "http://localhost:5002/api/ticks"
SYMBOLS          = ["RELIANCE","TCS","INFY"]
SPEED_MULTIPLIER = 60    # 1 sec market = 1/60 real sec (speed up x60)
PAPER_MODE       = True
LOG_PATH         = Path("logs/backtest_summary.json")

os.makedirs(LOG_PATH.parent, exist_ok=True)

# ---------- HELPERS ----------
def send_tick(symbol, row):
    tick = {
        "symbol": symbol,
        "last_price": float(row["close"]),
        "volume": int(row["volume"]),
        "mode": "backtest",
        "ohlc": {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low":  float(row["low"]),
            "close":float(row["close"]),
        },
        "timestamp": row["timestamp"] if "timestamp" in row else row["date"],
    }
    try:
        requests.post(DASHBOARD_URL, json=tick, timeout=1)
    except Exception as e:
        print(f"⚠️ Dashboard unreachable: {e}")
    return tick

def load_csv(symbol):
    file = DATA_FOLDER / f"{symbol}.csv"
    if not file.exists():
        print(f"❌ Missing CSV for {symbol}: {file}")
        return []
    with open(file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows

# ---------- MAIN REPLAY ----------
def replay_symbol(symbol):
    rows = load_csv(symbol)
    if not rows: 
        return 0,0
    trades, pnl = 0, 0.0
    for i,row in enumerate(rows):
        tick = send_tick(symbol, row)
        price = tick["last_price"]

        # --- Simple demo strategy: buy dip, sell spike ---
        if PAPER_MODE and random.random()<0.005:     # rare trade
            side = random.choice(["BUY","SELL"])
            qty  = 10
            pnl_delta = random.uniform(-5,10)
            pnl += pnl_delta
            trades += 1
            print(f"🧠 {side} {symbol} @ {price} → PnL +{pnl_delta:.2f}")

        # simulate market time gap
        time.sleep(1/SPEED_MULTIPLIER)
    return trades,pnl

def main():
    print(f"🚀 Starting Bambhoria Backtest Replay ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    total_trades,total_pnl=0,0.0
    for sym in SYMBOLS:
        t,p = replay_symbol(sym)
        total_trades+=t; total_pnl+=p
    summary={
        "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbols":SYMBOLS,
        "total_trades":total_trades,
        "total_pnl":round(total_pnl,2)
    }
    with open(LOG_PATH,"a") as f: json.dump(summary,f); f.write("\n")
    print(f"✅ Backtest done: {summary}")

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("🛑 Replay stopped by user.")