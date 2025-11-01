"""
bambhoria_godeye_v51_fusion.py
Bambhoria God-Eye V51.0 — Quantum Neural Fusion Upgrade
Author:  Vikas Bambhoria
Purpose:
 - Single-click orchestrator for all Bambhoria subsystems
 - Launch, monitor, auto-heal, and report
"""

import subprocess, threading, time, psutil, json, os
from datetime import datetime
from pathlib import Path

MODULES = {
    "Dashboard" : "python dashboard_server.py",
    "Feed"      : "python http_mock_feed.py",
    "Signal"    : "python adaptive_signal_engine.py",
    "Monitor"   : "python quantum_performance_monitor.py",
    "AutoHeal"  : "python launch_auto_heal.py",
    "Insight"   : "python neural_insight_engine.py"
}
LOG_DIR = Path("logs")
os.makedirs(LOG_DIR, exist_ok=True)
HEARTBEAT_FILE = LOG_DIR / "fusion_heartbeat.json"

# Track spawned processes for cleanup
spawned_processes = []

# ---------- Launchers ----------
def start_process(name, cmd):
    try:
        proc = subprocess.Popen(cmd, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
        spawned_processes.append({"name": name, "process": proc, "cmd": cmd})
        print(f"[Fusion] 🚀 {name} started (PID: {proc.pid}).")
        time.sleep(2)  # Give process time to initialize
    except Exception as e:
        print(f"[Fusion] ❌ Failed to start {name}: {e}")

def module_alive(keyword):
    """Check if a module process is running by looking for its script name in cmdline."""
    for p in psutil.process_iter(["cmdline"]):
        try:
            cmdline = p.info.get("cmdline")
            if cmdline and isinstance(cmdline, list):
                cmdline_str = " ".join(cmdline)
                # Look for the script name (e.g., "dashboard_server.py")
                if keyword in cmdline_str:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

# ---------- Watchdog ----------
def heartbeat_loop():
    """Monitor all modules and restart any that have crashed."""
    restart_count = {}
    max_restarts = 5
    
    while True:
        hb = {"time": datetime.now().isoformat(), "modules": {}}
        for name, cmd in MODULES.items():
            # Extract script name for process detection
            script_name = cmd.split()[1]  # e.g., "dashboard_server.py"
            running = module_alive(script_name)
            hb["modules"][name] = "OK" if running else "DOWN"
            
            if not running:
                # Check restart count to prevent infinite restart loops
                restart_count[name] = restart_count.get(name, 0) + 1
                if restart_count[name] <= max_restarts:
                    print(f"[Fusion] ⚠️  {name} down → restarting... (attempt {restart_count[name]}/{max_restarts})")
                    start_process(name, cmd)
                else:
                    print(f"[Fusion] ❌ {name} exceeded max restart attempts ({max_restarts}), giving up.")
            else:
                # Reset restart count if module is running
                if name in restart_count:
                    restart_count[name] = 0
                    
        with open(HEARTBEAT_FILE, "w") as f: 
            json.dump(hb, f, indent=2)
        time.sleep(15)  # Check every 15 seconds

# ---------- Startup ----------
def cleanup():
    """Terminate all spawned processes."""
    print("\n[Fusion] 🛑 Shutting down all modules...")
    for item in spawned_processes:
        try:
            name = item["name"]
            proc = item["process"]
            if proc.poll() is None:  # Process still running
                print(f"[Fusion] Terminating {name} (PID: {proc.pid})...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"[Fusion] Force killing {name}...")
                    proc.kill()
        except Exception as e:
            print(f"[Fusion] Error cleaning up {item.get('name', 'unknown')}: {e}")
    print("[Fusion] ✅ All modules stopped.")

def main():
    print("🧠 Bambhoria God-Eye V51.0 — Quantum Neural Fusion Upgrade")
    print("----------------------------------------------------------")
    for name,cmd in MODULES.items(): start_process(name, cmd)
    print("✅ All modules launched. Monitoring active...")
    print("💡 Press Ctrl+C to stop all modules and exit.\n")
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()

if __name__=="__main__":
    main()
