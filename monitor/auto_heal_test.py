"""
Auto-Heal Engine Demo & Test
Tests the auto-heal functionality with a controlled scenario
"""

import subprocess
import time
import requests
import psutil

def test_auto_heal():
    """Test the auto-heal engine functionality"""
    
    print("🧪 Auto-Heal Engine Test")
    print("=" * 50)
    
    # Step 1: Start the main dashboard server
    print("1️⃣ Starting main dashboard server...")
    dashboard_process = subprocess.Popen(
        ["python", "../dashboard_server.py"],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    
    # Wait for startup
    time.sleep(3)
    
    # Step 2: Test health endpoint
    print("2️⃣ Testing health endpoint...")
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ Health endpoint working: {response.json()}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
    
    # Step 3: Start auto-heal engine
    print("3️⃣ Starting auto-heal engine...")
    print("💡 Check the new console window for auto-heal logs")
    
    autoheal_process = subprocess.Popen(
        ["python", "auto_heal_engine.py"],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    
    print("4️⃣ Auto-heal engine is now monitoring services...")
    print("📊 You should see periodic health checks in the auto-heal console")
    print("🔧 To test recovery: manually close a service window to see auto-restart")
    print("🛑 Press Ctrl+C here to stop the test")
    
    try:
        # Keep test running
        while True:
            time.sleep(5)
            print("⏰ Test running... (Press Ctrl+C to stop)")
    except KeyboardInterrupt:
        print("\n🛑 Stopping test...")
        
        # Clean up processes
        dashboard_process.terminate()
        autoheal_process.terminate()
        
        print("✅ Test completed")

if __name__ == "__main__":
    test_auto_heal()