"""
order_control.py
Bambhoria Order Control & Execution Brain v1.0
Author: Vikas Bambhoria
Purpose:
 - Central trade engine controlling order flow
 - Integrates RiskManager + Mock / Live order execution
 - Updates logs & dashboard after each trade
"""

import json, os, time, random, requests
from datetime import datetime
from .risk_manager import RiskManager

# -------- CONFIG --------
DASHBOARD_ORDER_ENDPOINT = "http://localhost:5006/api/orders"
ORDER_LOG_PATH = "logs/orders_log.json"
MODE = "paper"   # "paper" or "live"

os.makedirs(os.path.dirname(ORDER_LOG_PATH), exist_ok=True)
risk = RiskManager(max_daily_loss=5000, max_position_size=100)

# -------- HELPERS --------
def _log(msg):
    print(f"[OrderBrain] {msg}")

def _write_json(obj):
    with open(ORDER_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

# -------- MAIN CLASS --------
class OrderBrain:
    def __init__(self, mode=MODE):
        self.mode = mode
        _log(f"Initialized in {self.mode.upper()} mode")

    def simulate_trade(self, symbol, side, qty, price):
        """Paper trade simulation."""
        pnl = round(random.uniform(-3.0, 8.0), 2) * qty
        exec_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trade = {
            "symbol": symbol, "side": side, "qty": qty,
            "price": price, "pnl": pnl, "mode": self.mode,
            "time": exec_time
        }
        risk.record_trade(symbol, qty, price, side, pnl)
        _write_json(trade)
        _log(f"🧠 Simulated {side} {symbol} x{qty} @ {price} → PnL {pnl:+}")
        try:
            requests.post(DASHBOARD_ORDER_ENDPOINT, json=trade, timeout=2)
        except Exception as e:
            _log(f"⚠️ Dashboard order post failed: {e}")
        return trade

    def place_order(self, symbol, side, qty, price, volatility=10):
        """Main entry for any order."""
        risk_check = risk.check_risk(symbol, qty, price, side)
        if not risk_check['allowed']:
            _log(f"🚫 Order blocked by RiskManager: {risk_check['reason']}")
            return {"status":"blocked","reason":risk_check['reason']}

        if self.mode == "paper":
            return self.simulate_trade(symbol, side, qty, price)
        else:
            # Future: integrate Zerodha connector here
            _log(f"⚙️ Live order placeholder for {symbol}")
            return {"status":"live_pending"}

# -------- DEMO --------
if __name__ == "__main__":
    brain = OrderBrain(mode="paper")
    demo_orders = [
        ("RELIANCE","BUY",10,2550),
        ("RELIANCE","SELL",10,2560),
        ("TCS","BUY",5,3850)
    ]
    for sym,side,qty,price in demo_orders:
        brain.place_order(sym,side,qty,price)
        time.sleep(0.5)
    _log("Summary:")
    risk.summary()