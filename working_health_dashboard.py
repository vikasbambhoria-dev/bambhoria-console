"""
Working Health Dashboard Test
Based on successful dashboard pattern
"""
from flask import Flask, jsonify, render_template_string
import psutil
import time
import json
from datetime import datetime

app = Flask(__name__)

# Simple data storage
app_data = {
    "status": "healthy",
    "uptime": datetime.now().isoformat(),
    "requests": 0
}

@app.route("/")
def home():
    app_data["requests"] += 1
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head><title>Health Test Dashboard</title></head>
    <body>
        <h1>🌟 Health Test Dashboard</h1>
        <p>Status: <span id="status">{{ status }}</span></p>
        <p>Uptime: {{ uptime }}</p>
        <p>Requests: {{ requests }}</p>
        <button onclick="testHealth()">Test Health Endpoint</button>
        <div id="health-result"></div>
        
        <script>
        async function testHealth() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                document.getElementById('health-result').innerHTML = 
                    '<h3>Health Response:</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
            } catch (error) {
                document.getElementById('health-result').innerHTML = 
                    '<h3>Error:</h3><p>' + error.message + '</p>';
            }
        }
        </script>
    </body>
    </html>
    """, **app_data)

@app.route("/api/health")
def api_health():
    """Health endpoint with error handling"""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        health_data = {
            "status": "healthy",
            "cpu": round(cpu_percent, 1),
            "mem": round(memory.percent, 1),
            "time": time.strftime("%H:%M:%S"),
            "timestamp": datetime.now().isoformat()
        }
        
        app_data["requests"] += 1
        return jsonify(health_data)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "time": time.strftime("%H:%M:%S")
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Working Health Dashboard on http://localhost:5013")
    app.run(host='127.0.0.1', port=5013, debug=False)