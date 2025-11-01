"""
Simple test server to validate health endpoint functionality
"""
from flask import Flask, jsonify
import psutil
import time

app = Flask(__name__)

@app.route('/api/health')
def api_health():
    return jsonify({
        "cpu": psutil.cpu_percent(),
        "mem": psutil.virtual_memory().percent,
        "time": time.strftime("%H:%M:%S")
    })

if __name__ == '__main__':
    print("🔧 Test Health Server starting on port 5010")
    app.run(host='127.0.0.1', port=5010, debug=True)