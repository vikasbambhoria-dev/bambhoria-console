"""
system_monitor_dashboard.py
Bambhoria System Monitor & Visualization Dashboard
Real-time monitoring of the complete trading architecture
"""

from flask import Flask, render_template_string, jsonify, request
import json
import time
from datetime import datetime

app = Flask(__name__)

# Global system state
system_state = {
    "components": {
        "feed_source": {"status": "Unknown", "last_update": None, "metrics": {}},
        "signal_generator": {"status": "Unknown", "last_update": None, "metrics": {}},
        "order_brain": {"status": "Unknown", "last_update": None, "metrics": {}},
        "risk_manager": {"status": "Unknown", "last_update": None, "metrics": {}},
        "dashboard": {"status": "Active", "last_update": datetime.now().isoformat(), "metrics": {}}
    },
    "system_stats": {
        "feeds_processed": 0,
        "signals_generated": 0,
        "orders_placed": 0,
        "risk_blocks": 0,
        "total_pnl": 0.0,
        "uptime": 0
    },
    "alerts": [],
    "last_update": datetime.now().isoformat()
}

MONITOR_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Bambhoria System Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #0d1117; color: #f0f6fc; }
        .header { text-align: center; margin-bottom: 30px; }
        .architecture { text-align: center; margin: 20px 0; padding: 20px; background: #161b22; border-radius: 10px; }
        .flow-diagram { font-family: monospace; font-size: 14px; line-height: 1.8; }
        .components { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }
        .component { background: #21262d; border: 2px solid #30363d; border-radius: 10px; padding: 20px; }
        .component.active { border-color: #238636; box-shadow: 0 0 10px rgba(35,134,54,0.3); }
        .component.error { border-color: #da3633; box-shadow: 0 0 10px rgba(218,54,51,0.3); }
        .component.warning { border-color: #f85149; box-shadow: 0 0 10px rgba(248,81,73,0.3); }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }
        .stat-box { background: #21262d; border-radius: 10px; padding: 20px; text-align: center; border: 1px solid #30363d; }
        .stat-value { font-size: 2em; font-weight: bold; color: #238636; }
        .alerts { background: #21262d; border-radius: 10px; padding: 20px; margin: 20px 0; max-height: 400px; overflow-y: auto; }
        .alert { padding: 10px; border-bottom: 1px solid #30363d; font-family: monospace; }
        .alert:last-child { border-bottom: none; }
        .alert.info { color: #79c0ff; }
        .alert.warning { color: #f85149; }
        .alert.error { color: #da3633; }
        .status { font-weight: bold; }
        .status.active { color: #238636; }
        .status.error { color: #da3633; }
        .status.warning { color: #f85149; }
        .status.unknown { color: #f0f6fc; }
        .metric { font-size: 0.9em; color: #8b949e; margin: 5px 0; }
    </style>
    <script>
        function updateMonitor() {
            fetch('/api/system_status')
                .then(response => response.json())
                .then(data => {
                    // Update component statuses
                    Object.keys(data.components).forEach(comp => {
                        const component = data.components[comp];
                        const statusEl = document.getElementById(comp + '-status');
                        const updateEl = document.getElementById(comp + '-update');
                        
                        if (statusEl) statusEl.textContent = component.status;
                        if (updateEl) updateEl.textContent = component.last_update || 'Never';
                        
                        // Update component styling
                        const compEl = document.getElementById(comp + '-component');
                        if (compEl) {
                            compEl.className = 'component';
                            if (component.status === 'Active') compEl.classList.add('active');
                            else if (component.status === 'Error') compEl.classList.add('error');
                            else if (component.status === 'Warning') compEl.classList.add('warning');
                        }
                    });
                    
                    // Update system stats
                    document.getElementById('feeds-processed').textContent = data.system_stats.feeds_processed;
                    document.getElementById('signals-generated').textContent = data.system_stats.signals_generated;
                    document.getElementById('orders-placed').textContent = data.system_stats.orders_placed;
                    document.getElementById('risk-blocks').textContent = data.system_stats.risk_blocks;
                    document.getElementById('total-pnl').textContent = '₹' + data.system_stats.total_pnl.toFixed(2);
                    document.getElementById('uptime').textContent = Math.floor(data.system_stats.uptime / 60) + 'm ' + Math.floor(data.system_stats.uptime % 60) + 's';
                    
                    // Update alerts
                    const alertsDiv = document.getElementById('alerts');
                    alertsDiv.innerHTML = data.alerts.slice(-20).reverse().map(alert => 
                        `<div class="alert ${alert.level.toLowerCase()}">
                            [${new Date(alert.timestamp).toLocaleTimeString()}] ${alert.message}
                        </div>`
                    ).join('');
                    
                    document.getElementById('last-update').textContent = new Date(data.last_update).toLocaleTimeString();
                })
                .catch(error => console.error('Update failed:', error));
        }
        
        setInterval(updateMonitor, 2000);
        updateMonitor();
    </script>
</head>
<body>
    <div class="header">
        <h1>🎯 Bambhoria System Monitor</h1>
        <p>Real-time monitoring of complete trading architecture</p>
        <p><small>Last Update: <span id="last-update">--:--:--</span></small></p>
    </div>
    
    <div class="architecture">
        <h3>📊 Trading System Architecture</h3>
        <div class="flow-diagram">
Mock/Live Feed → Signal Generator → Order Brain → Risk Manager → Dashboard<br>
                                           ↘<br>
                                            ↘ Logs/PnL/Alerts → Visualization
        </div>
    </div>
    
    <div class="components">
        <div class="component" id="feed_source-component">
            <h3>📡 Feed Source</h3>
            <div class="status" id="feed_source-status">Unknown</div>
            <div class="metric">Last Update: <span id="feed_source-update">Never</span></div>
        </div>
        <div class="component" id="signal_generator-component">
            <h3>🧠 Signal Generator</h3>
            <div class="status" id="signal_generator-status">Unknown</div>
            <div class="metric">Last Update: <span id="signal_generator-update">Never</span></div>
        </div>
        <div class="component" id="order_brain-component">
            <h3>⚡ Order Brain</h3>
            <div class="status" id="order_brain-status">Unknown</div>
            <div class="metric">Last Update: <span id="order_brain-update">Never</span></div>
        </div>
        <div class="component" id="risk_manager-component">
            <h3>🛡️ Risk Manager</h3>
            <div class="status" id="risk_manager-status">Unknown</div>
            <div class="metric">Last Update: <span id="risk_manager-update">Never</span></div>
        </div>
        <div class="component active" id="dashboard-component">
            <h3>📈 Dashboard</h3>
            <div class="status active" id="dashboard-status">Active</div>
            <div class="metric">Last Update: <span id="dashboard-update">Now</span></div>
        </div>
    </div>
    
    <div class="stats">
        <div class="stat-box">
            <div class="stat-value" id="feeds-processed">0</div>
            <div>Feeds Processed</div>
        </div>
        <div class="stat-box">
            <div class="stat-value" id="signals-generated">0</div>
            <div>Signals Generated</div>
        </div>
        <div class="stat-box">
            <div class="stat-value" id="orders-placed">0</div>
            <div>Orders Placed</div>
        </div>
        <div class="stat-box">
            <div class="stat-value" id="risk-blocks">0</div>
            <div>Risk Blocks</div>
        </div>
        <div class="stat-box">
            <div class="stat-value" id="total-pnl">₹0.00</div>
            <div>Total P&L</div>
        </div>
        <div class="stat-box">
            <div class="stat-value" id="uptime">0s</div>
            <div>System Uptime</div>
        </div>
    </div>
    
    <div class="alerts">
        <h3>🚨 System Alerts & Logs</h3>
        <div id="alerts">
            <div class="alert info">System monitor initialized...</div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def monitor_dashboard():
    return render_template_string(MONITOR_HTML)

@app.route('/api/system_status')
def get_system_status():
    return jsonify(system_state)

@app.route('/api/system_update', methods=['POST'])
def system_update():
    """Receive system updates from the trading system"""
    try:
        data = request.json
        if data:
            # Update system stats
            if 'stats' in data:
                system_state['system_stats'].update(data['stats'])
            
            # Update component status
            if 'component_status' in data:
                for comp, status in data['component_status'].items():
                    if comp in system_state['components']:
                        system_state['components'][comp]['status'] = status
                        system_state['components'][comp]['last_update'] = datetime.now().isoformat()
            
            # Update alerts
            if 'alerts' in data:
                system_state['alerts'] = data['alerts']
            
            system_state['last_update'] = datetime.now().isoformat()
        
        return jsonify({"status": "received"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/component_update/<component>', methods=['POST'])
def component_update(component):
    """Update specific component status"""
    try:
        data = request.json
        if component in system_state['components']:
            system_state['components'][component].update(data)
            system_state['components'][component]['last_update'] = datetime.now().isoformat()
            system_state['last_update'] = datetime.now().isoformat()
        return jsonify({"status": "received"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health')
def api_health():
    import psutil, time
    return jsonify({
        "cpu": psutil.cpu_percent(),
        "mem": psutil.virtual_memory().percent,
        "time": time.strftime("%H:%M:%S")
    })

if __name__ == '__main__':
    print("🎯 Starting Bambhoria System Monitor Dashboard on http://localhost:5008")
    app.run(host='0.0.0.0', port=5008, debug=False)