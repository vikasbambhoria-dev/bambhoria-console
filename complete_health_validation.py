"""
Complete Health Endpoint Validation
Tests all main dashboard health endpoints
"""

import subprocess
import time
import requests
import sys
import json

# Dashboard configurations
DASHBOARDS = {
    "pipeline": {
        "file": "pipeline_dashboard.py",
        "port": 5004,
        "name": "Pipeline Dashboard"
    },
    "main": {
        "file": "dashboard_server.py", 
        "port": 5000,
        "name": "Main Dashboard"
    },
    "system": {
        "file": "system_monitor_dashboard.py",
        "port": 5008, 
        "name": "System Monitor"
    },
    "performance": {
        "file": "performance_dashboard.py",
        "port": 5009,
        "name": "Performance Dashboard"
    }
}

def test_health_endpoint(port, name):
    """Test individual health endpoint"""
    try:
        url = f"http://localhost:{port}/api/health"
        response = requests.get(url, timeout=3)
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ {name} (:{port}) - Health: {health_data}")
            return True
        else:
            print(f"❌ {name} (:{port}) - Status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"🔌 {name} (:{port}) - Server not running")
        return False
    except Exception as e:
        print(f"❌ {name} (:{port}) - Error: {e}")
        return False

def start_dashboard(file_name, port, name):
    """Start a dashboard server"""
    try:
        print(f"🚀 Starting {name} on port {port}...")
        process = subprocess.Popen(
            [sys.executable, file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(2)  # Wait for startup
        return process
    except Exception as e:
        print(f"❌ Failed to start {name}: {e}")
        return None

def main():
    """Main validation function"""
    print("🎯 Bambhoria Health Endpoint Validation")
    print("=" * 50)
    
    # Test existing running servers first
    print("\n🔍 Testing currently running servers...")
    running_count = 0
    for key, config in DASHBOARDS.items():
        if test_health_endpoint(config["port"], config["name"]):
            running_count += 1
    
    print(f"\n📊 Summary: {running_count}/{len(DASHBOARDS)} health endpoints working")
    
    if running_count < len(DASHBOARDS):
        print("\n🔧 Starting missing dashboards...")
        processes = []
        
        for key, config in DASHBOARDS.items():
            # Try to start the dashboard
            process = start_dashboard(config["file"], config["port"], config["name"])
            if process:
                processes.append((process, config))
        
        # Test all health endpoints after startup
        print("\n🔍 Testing health endpoints after startup...")
        working_count = 0
        for key, config in DASHBOARDS.items():
            if test_health_endpoint(config["port"], config["name"]):
                working_count += 1
        
        print(f"\n🎉 Final Result: {working_count}/{len(DASHBOARDS)} health endpoints working!")
        
        # Cleanup processes (optional - they can keep running)
        print("\n🧹 Dashboard servers are running in background...")
        print("💡 Use Ctrl+C in their terminals to stop them when done")
    
    return running_count == len(DASHBOARDS)

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ ALL HEALTH ENDPOINTS WORKING!")
    else:
        print("\n⚠️  Some health endpoints need attention")