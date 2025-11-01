"""
Ultra Simple Health Endpoint Test
"""
from flask import Flask, jsonify
import time

app = Flask(__name__)

@app.route('/')
def home():
    return "Server is running!"

@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "time": time.strftime("%H:%M:%S"),
        "message": "Server is operational"
    })

if __name__ == '__main__':
    print("🌟 Starting ultra simple health server on port 5012")
    app.run(host='127.0.0.1', port=5012, debug=False)