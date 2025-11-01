"""
quantum_performance_monitor.py
Bambhoria Quantum Performance Monitor v1.0
Author: Vikas Bambhoria
Purpose:
 - Monitor real-time system health for Bambhoria God-Eye
 - Track CPU, memory, latency, tick-rate, trade success, and warnings
"""

import os, psutil, time, json, requests, statistics
from datetime import datetime, timedelta
from pathlib import Path

MONITOR_LOG = Path("logs/performance_monitor.json")
DASHBOARD_STATUS_URL = "http://localhost:5000/api/pnl"
PING_INTERVAL = 5
LATENCY_SAMPLES = []
TICK_RATE = []
TRADES_SUCCESS, TRADES_FAILED = 0, 0

os.makedirs(MONITOR_LOG.parent, exist_ok=True)

def _log(msg):
    print(f"[Monitor] {msg}")
    with open(MONITOR_LOG, "a") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(), "msg": msg}) + "\n")

def get_latency():
    """Ping dashboard endpoint and return latency ms."""
    t0 = time.time()
    try:
        resp = requests.get(DASHBOARD_STATUS_URL, timeout=2)
        if resp.ok:
            latency = (time.time() - t0) * 1000
            LATENCY_SAMPLES.append(latency)
            return round(latency, 2)
    except Exception:
        pass
    return None

def monitor_system():
    _log("🚀 Quantum Performance Monitor started.")
    start_time = datetime.now()
    while True:
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        latency = get_latency()
        tick_rate = round(statistics.mean(TICK_RATE[-5:]),2) if len(TICK_RATE)>=5 else 0
        trade_total = TRADES_SUCCESS + TRADES_FAILED
        trade_success_rate = round((TRADES_SUCCESS / trade_total)*100,2) if trade_total else 100

        status = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "cpu": cpu,
            "mem": mem,
            "latency_ms": latency or 0,
            "tick_rate": tick_rate,
            "trade_success_rate": trade_success_rate,
            "uptime_min": round((datetime.now()-start_time).total_seconds()/60,2)
        }
        _log(status)

        # ⚠️ alerts
        if cpu > 85: _log("⚠️ High CPU load detected!")
        if mem > 90: _log("⚠️ Memory pressure!")
        if latency and latency > 800: _log("⚠️ High latency!")
        time.sleep(PING_INTERVAL)

if __name__ == "__main__":
    try:
        monitor_system()
    except KeyboardInterrupt:
        _log("🛑 Monitor stopped by user.")