"""
Test all running ports for health endpoints
"""
import requests
import time

def test_port(port):
    try:
        # Test basic connectivity
        base_url = f"http://localhost:{port}"
        r = requests.get(base_url, timeout=2)
        print(f"Port {port}: Base response {r.status_code}")
        
        # Test health endpoint
        health_url = f"{base_url}/api/health"
        r = requests.get(health_url, timeout=2)
        if r.status_code == 200:
            print(f"Port {port}: ✅ Health endpoint working! Response: {r.json()}")
            return True
        else:
            print(f"Port {port}: ❌ Health endpoint returned {r.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"Port {port}: ❌ Connection refused")
        return False
    except Exception as e:
        print(f"Port {port}: ❌ Error: {e}")
        return False

# Test known running ports
running_ports = [5000, 5001, 5002, 5003, 5004, 5007, 5040]

print("🔍 Testing all running ports for health endpoints...")
print("=" * 60)

working_health_endpoints = []
for port in running_ports:
    if test_port(port):
        working_health_endpoints.append(port)
    print("-" * 40)

print(f"\n🎯 Summary: Found {len(working_health_endpoints)} working health endpoints on ports: {working_health_endpoints}")