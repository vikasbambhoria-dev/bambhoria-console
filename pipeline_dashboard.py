"""
pipeline_dashboard.py
Bambhoria Pipeline Monitoring Dashboard
Real-time view of the complete trading pipeline
"""

from flask import Flask, render_template_string, jsonify, request
import json
from datetime import datetime

app = Flask(__name__)

# Global pipeline state
pipeline_state = {
    "feed_status": "Unknown",
    "signal_generator_status": "Unknown",
    "risk_manager_status": "Unknown", 
    "order_brain_status": "Unknown",
    "dashboard_status": "Active",
    "last_update": datetime.now().isoformat(),
    "stats": {
        "total_signals": 0,
        "total_trades": 0,
        "total_pnl": 0.0,
        "uptime": "0:00:00"
    },
    "recent_activities": []
}

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Bambhoria Trading Pipeline Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: #fff; }
        .header { text-align: center; margin-bottom: 30px; }
        .pipeline-flow { display: flex; justify-content: center; align-items: center; margin: 20px 0; flex-wrap: wrap; }
        .component { background: #2d2d2d; border: 2px solid #4a4a4a; border-radius: 10px; padding: 15px; margin: 10px; min-width: 150px; text-align: center; }
        .component.active { border-color: #00ff00; box-shadow: 0 0 10px rgba(0,255,0,0.3); }
        .component.error { border-color: #ff4444; box-shadow: 0 0 10px rgba(255,68,68,0.3); }
        .arrow { font-size: 24px; color: #4a4a4a; margin: 0 10px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }
        .stat-box { background: #2d2d2d; border-radius: 10px; padding: 20px; text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; color: #00ff00; }
        .activities { background: #2d2d2d; border-radius: 10px; padding: 20px; margin: 20px 0; max-height: 300px; overflow-y: auto; }
        .activity { padding: 10px; border-bottom: 1px solid #4a4a4a; font-family: monospace; }
        .activity:last-child { border-bottom: none; }
        .status { font-weight: bold; }
        .status.active { color: #00ff00; }
        .status.error { color: #ff4444; }
        .status.unknown { color: #ffaa00; }
    </style>
    <script>
        function updateDashboard() {
            fetch('/api/pipeline_status')
                .then(response => response.json())
                .then(data => {
                    // Update component statuses
                    document.getElementById('feed-status').textContent = data.feed_status;
                    document.getElementById('signal-status').textContent = data.signal_generator_status;
                    document.getElementById('risk-status').textContent = data.risk_manager_status;
                    document.getElementById('order-status').textContent = data.order_brain_status;
                    
                    // Update stats
                    document.getElementById('total-signals').textContent = data.stats.total_signals;
                    document.getElementById('total-trades').textContent = data.stats.total_trades;
                    document.getElementById('total-pnl').textContent = '₹' + data.stats.total_pnl.toFixed(2);
                    document.getElementById('uptime').textContent = data.stats.uptime;
                    
                    // Update activities
                    const activitiesDiv = document.getElementById('activities');
                    activitiesDiv.innerHTML = data.recent_activities.map(activity => 
                        `<div class="activity">${activity}</div>`
                    ).join('');
                    
                    document.getElementById('last-update').textContent = new Date(data.last_update).toLocaleTimeString();
                })
                .catch(error => console.error('Update failed:', error));
        }
        
        setInterval(updateDashboard, 2000);
        updateDashboard();
    </script>
</head>
<body>
    <div class="header">
        <h1>🚀 Bambhoria Trading Pipeline Monitor</h1>
        <p>Real-time monitoring of the complete trading system</p>
        <p><small>Last Update: <span id="last-update">--:--:--</span></small></p>
    </div>
    
    <div class="pipeline-flow">
        <div class="component">
            <h3>📊 Feed Source</h3>
            <div class="status" id="feed-status">Unknown</div>
        </div>
        <div class="arrow">→</div>
        <div class="component">
            <h3>🧠 Signal Generator</h3>
            <div class="status" id="signal-status">Unknown</div>
        </div>
        <div class="arrow">→</div>
        <div class="component">
            <h3>🛡️ Risk Manager</h3>
            <div class="status" id="risk-status">Unknown</div>
        </div>
        <div class="arrow">→</div>
        <div class="component">
            <h3>⚡ Order Brain</h3>
            <div class="status" id="order-status">Unknown</div>
        </div>
        <div class="arrow">→</div>
        <div class="component active">
            <h3>📈 Dashboard</h3>
            <div class="status active">Active</div>
        </div>
    </div>
    
    <div class="stats">
        <div class="stat-box">
            <div class="stat-value" id="total-signals">0</div>
            <div>Total Signals</div>
        </div>
        <div class="stat-box">
            <div class="stat-value" id="total-trades">0</div>
            <div>Total Trades</div>
        </div>
        <div class="stat-box">
            <div class="stat-value" id="total-pnl">₹0.00</div>
            <div>Total P&L</div>
        </div>
        <div class="stat-box">
            <div class="stat-value" id="uptime">0:00:00</div>
            <div>Uptime</div>
        </div>
    </div>
    
    <div class="activities">
        <h3>📋 Recent Activities</h3>
        <div id="activities">
            <div class="activity">Waiting for pipeline data...</div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/pipeline_status')
def get_pipeline_status():
    return jsonify(pipeline_state)

@app.route('/api/pipeline_update', methods=['POST'])
def pipeline_update():
    # Endpoint for pipeline components to send updates
    return jsonify({"status": "received"})

@app.route("/api/signals", methods=["POST"])
def api_signals():
    sig = request.json
    print("📡 Signal:", sig)
    
    # Update pipeline state with signal data
    if sig:
        pipeline_state["signal_generator_status"] = "Active"
        pipeline_state["last_update"] = datetime.now().isoformat()
        pipeline_state["stats"]["total_signals"] += 1
        
        # Add to recent activities
        activity = f"[{datetime.now().strftime('%H:%M:%S')}] Signal: {sig}"
        pipeline_state["recent_activities"].insert(0, activity)
        
        # Keep only last 10 activities
        if len(pipeline_state["recent_activities"]) > 10:
            pipeline_state["recent_activities"] = pipeline_state["recent_activities"][:10]
    
    return jsonify({"ok": True})

@app.route("/api/health")
def api_health():
    import psutil, time
    return jsonify({
        "cpu": psutil.cpu_percent(),
        "mem": psutil.virtual_memory().percent,
        "time": time.strftime("%H:%M:%S")
    })

if __name__ == '__main__':
    print("🎯 Starting Pipeline Dashboard on http://localhost:5004")
    app.run(host='0.0.0.0', port=5004, debug=False)