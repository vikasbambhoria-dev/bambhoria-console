"""
Asynchronous Health Endpoint Test
"""
import subprocess
import time
import requests
import sys
import os

def test_health_endpoint():
    """Test health endpoint by starting server and testing it"""
    
    # Start the working health dashboard in background
    print("🚀 Starting health dashboard...")
    server_process = subprocess.Popen([
        sys.executable, "working_health_dashboard.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    
    try:
        # Test the health endpoint
        print("🔍 Testing health endpoint...")
        response = requests.get("http://localhost:5013/api/health", timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health endpoint working!")
            print(f"📊 Health Data: {health_data}")
            return True
        else:
            print(f"❌ Health endpoint returned status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to health endpoint")
        return False
    except Exception as e:
        print(f"❌ Error testing health endpoint: {e}")
        return False
    finally:
        # Clean up
        print("🧹 Cleaning up server process...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()

if __name__ == "__main__":
    success = test_health_endpoint()
    if success:
        print("\n🎯 SUCCESS: Health endpoint integration is working!")
    else:
        print("\n❌ FAILED: Health endpoint needs debugging")