"""
auto_heal_engine.py
Bambhoria Auto-Heal & Recovery Engine v1.0
Author:  Vikas Bambhoria
Purpose:
 - Watch all critical services (Dashboard, Feed, Signal, Monitor)
 - Auto-restart crashed processes
 - Maintain recovery logs
"""

import os, psutil, time, subprocess, json, requests
from datetime import datetime, timedelta
from pathlib import Path

# ---------- CONFIG ----------
SERVICES = {
    "dashboard_server.py": "python ../dashboard/dashboard_server.py",
    "advanced_mock_feed.py": "python ../advanced_mock_feed.py",
    "signal_generator.py": "python ../signal_generator.py",
}
HEALTH_URL  = "http://localhost:5000/api/health"   # dashboard health check
CHECK_DELAY = 10                                # sec between checks
LOG_PATH    = Path("../logs/auto_heal_log.json")
os.makedirs(LOG_PATH.parent, exist_ok=True)

# ---------- HELPERS ----------
def _log(msg):
    entry = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "msg": msg}
    print(f"[Auto-Heal] {entry['time']}  {msg}")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def _process_running(keyword):
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            if any(keyword in ' '.join(p.info['cmdline']) for _ in [0]):
                return True
        except Exception:
            pass
    return False

def _restart_service(cmd):
    _log(f"Restarting → {cmd}")
    # Change to parent directory and run command
    full_cmd = f"cd .. && {cmd}"
    subprocess.Popen(full_cmd, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)

def _ping_dashboard():
    try:
        r = requests.get(HEALTH_URL, timeout=2)
        return r.ok
    except Exception:
        return False

# ---------- MAIN LOOP ----------
def auto_heal():
    _log("🩺  Auto-Heal Engine started.")
    while True:
        for file, cmd in SERVICES.items():
            if not _process_running(file):
                _log(f"⚠️  {file} not running → Restarting...")
                _restart_service(cmd)
            else:
                _log(f"✅  {file} OK")
        if not _ping_dashboard():
            _log("⚠️  Dashboard unresponsive → Restarting server...")
            _restart_service(SERVICES["dashboard_server.py"])
        time.sleep(CHECK_DELAY)

if __name__ == "__main__":
    try:
        auto_heal()
    except KeyboardInterrupt:
        _log("🛑 Auto-Heal stopped by user.")