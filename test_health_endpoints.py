"""
Test health API validation
"""
import requests
import time

def test_health_endpoint(port, endpoint="/api/health"):
    url = f"http://localhost:{port}{endpoint}"
    try:
        print(f"Testing: {url}")
        response = requests.get(url, timeout=5)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data}")
            return True
        else:
            print(f"Error response: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection refused - server not running on port {port}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    # Test all dashboard ports
    ports = [5006, 5008, 5009, 5010]
    for port in ports:
        print(f"\n🔍 Testing port {port}")
        test_health_endpoint(port)
        print("-" * 50)