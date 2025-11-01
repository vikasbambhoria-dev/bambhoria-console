"""
launch_complete_system.py
Complete Bambhoria Trading System Launcher
Starts all components in the correct order with proper coordination
"""

import subprocess
import time
import sys
import os
import requests
from threading import Thread

def start_component(script_name, port=None, delay=2):
    """Start a system component"""
    print(f"🚀 Starting {script_name}...")
    
    if port:
        # Check if port is already in use
        try:
            response = requests.get(f"http://localhost:{port}", timeout=1)
            print(f"⚠️  Port {port} already in use for {script_name}")
            return None
        except:
            pass  # Port is free
    
    process = subprocess.Popen([
        sys.executable, script_name
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    time.sleep(delay)
    return process

def check_component_health(name, url, timeout=5):
    """Check if a component is healthy"""
    try:
        response = requests.get(url, timeout=timeout)
        print(f"✅ {name} is healthy")
        return True
    except:
        print(f"❌ {name} not responding")
        return False

def main():
    print("🎯 Bambhoria Complete Trading System Launcher")
    print("=" * 60)
    
    # Change to the correct directory
    os.chdir("d:\\bambhoria\\godeye_v50_plus_auto_full_do_best")
    
    processes = []
    
    try:
        # 1. Start System Monitor Dashboard (Port 5008)
        monitor_process = start_component("system_monitor_dashboard.py", 5008, 3)
        if monitor_process:
            processes.append(("Monitor Dashboard", monitor_process))
        
        # 2. Start Analytics Dashboard (Port 5006)
        analytics_process = start_component("dashboard_server.py", 5006, 3)
        if analytics_process:
            processes.append(("Analytics Dashboard", analytics_process))
        
        # 3. Start Mock Feed Server (Port 8080)
        feed_process = start_component("http_mock_feed.py", 8080, 3)
        if feed_process:
            processes.append(("Mock Feed Server", feed_process))
        
        # 4. Start Complete Trading System
        trading_process = start_component("complete_trading_system.py", None, 5)
        if trading_process:
            processes.append(("Trading System", trading_process))
        
        print("\n🔍 System Health Check:")
        print("-" * 30)
        
        # Health checks
        time.sleep(5)
        check_component_health("System Monitor", "http://localhost:5008")
        check_component_health("Analytics Dashboard", "http://localhost:5006")
        check_component_health("Mock Feed Server", "http://localhost:8080")
        
        print(f"\n🎉 All systems launched! Running {len(processes)} processes")
        print("\n📊 Access your dashboards:")
        print("   • System Monitor:     http://localhost:5008")
        print("   • Analytics Dashboard: http://localhost:5006")
        print("   • Mock Feed Server:    http://localhost:8080")
        
        print("\n⚡ Complete Trading System Architecture Active!")
        print("   Mock/Live Feed → Signal Generator → Order Brain → Risk Manager → Dashboard")
        print("\n💡 Press Ctrl+C to shutdown all systems...")
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down all systems...")
        for name, process in processes:
            print(f"   Stopping {name}...")
            process.terminate()
        
        # Wait for clean shutdown
        time.sleep(2)
        print("✅ All systems stopped successfully!")

if __name__ == "__main__":
    main()