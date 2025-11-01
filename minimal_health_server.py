"""
Minimal Health Endpoint Test
"""
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Server is running"

@app.route('/api/health')
def health():
    try:
        import psutil
        import time
        
        # Get system metrics
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return jsonify({
            "status": "healthy",
            "cpu": round(cpu, 1),
            "mem": round(memory.percent, 1),
            "time": time.strftime("%H:%M:%S")
        })
    except Exception as e:
        return jsonify({
            "status": "error", 
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting minimal health server on port 5011")
    app.run(host='127.0.0.1', port=5011, debug=True)