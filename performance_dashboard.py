"""
performance_dashboard.py
Bambhoria Performance Dashboard for Quantum Monitor
Real-time system performance visualization
"""

from flask import Flask, render_template_string, jsonify
import json
import psutil
import time
from datetime import datetime
from pathlib import Path
import requests

app = Flask(__name__)

# Performance tracking
performance_data = {
    "current": {},
    "history": [],
    "alerts": [],
    "trading_status": {}
}

PERFORMANCE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Bambhoria Performance Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #0a0a0a; color: #00ff00; }
        .header { text-align: center; margin-bottom: 30px; }
        .title { color: #00ffff; font-size: 2.5em; text-shadow: 0 0 10px #00ffff; }
        .subtitle { color: #ffff00; margin-top: 10px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0; }
        .metric-card { 
            background: linear-gradient(135deg, #1a1a2e, #16213e); 
            border: 2px solid #00ff00; 
            border-radius: 15px; 
            padding: 20px; 
            box-shadow: 0 0 20px rgba(0,255,0,0.3);
        }
        .metric-title { color: #00ffff; font-size: 1.2em; margin-bottom: 15px; text-align: center; }
        .metric-value { font-size: 2.5em; font-weight: bold; text-align: center; margin: 10px 0; }
        .metric-value.normal { color: #00ff00; }
        .metric-value.warning { color: #ffff00; }
        .metric-value.critical { color: #ff0000; }
        .progress-bar { 
            width: 100%; 
            height: 20px; 
            background: #333; 
            border-radius: 10px; 
            overflow: hidden; 
            margin: 10px 0;
        }
        .progress-fill { 
            height: 100%; 
            transition: width 0.5s ease;
            border-radius: 10px;
        }
        .progress-fill.normal { background: linear-gradient(90deg, #00ff00, #00aa00); }
        .progress-fill.warning { background: linear-gradient(90deg, #ffff00, #aa6600); }
        .progress-fill.critical { background: linear-gradient(90deg, #ff0000, #aa0000); }
        .alerts { 
            background: #2a0a0a; 
            border: 2px solid #ff0000; 
            border-radius: 10px; 
            padding: 20px; 
            margin: 20px 0; 
            max-height: 300px; 
            overflow-y: auto;
        }
        .alert { 
            padding: 8px; 
            margin: 5px 0; 
            border-radius: 5px; 
            font-family: monospace;
        }
        .alert.critical { background: #4a0a0a; border-left: 4px solid #ff0000; }
        .alert.warning { background: #4a4a0a; border-left: 4px solid #ffff00; }
        .alert.info { background: #0a0a4a; border-left: 4px solid #00ffff; }
        .quantum-stats { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; 
            margin: 20px 0;
        }
        .stat-box { 
            background: #0a1a0a; 
            border: 1px solid #00ff00; 
            border-radius: 8px; 
            padding: 15px; 
            text-align: center;
        }
        .stat-label { color: #00ffff; font-size: 0.9em; }
        .stat-value { color: #00ff00; font-size: 1.8em; font-weight: bold; }
        .performance-score { 
            text-align: center; 
            margin: 30px 0;
            padding: 20px;
            background: radial-gradient(circle, #1a1a2e, #0a0a0a);
            border-radius: 15px;
            border: 3px solid #00ffff;
        }
        .score-title { color: #00ffff; font-size: 1.5em; margin-bottom: 15px; }
        .score-value { font-size: 4em; font-weight: bold; text-shadow: 0 0 20px; }
        .score-excellent { color: #00ff00; text-shadow: 0 0 20px #00ff00; }
        .score-good { color: #ffff00; text-shadow: 0 0 20px #ffff00; }
        .score-poor { color: #ff0000; text-shadow: 0 0 20px #ff0000; }
    </style>
    <script>
        function updateDashboard() {
            fetch('/api/performance')
                .then(response => response.json())
                .then(data => {
                    // Update performance score
                    const score = data.performance_score || 0;
                    const scoreElement = document.getElementById('performance-score');
                    scoreElement.textContent = score;
                    
                    // Set score color based on value
                    scoreElement.className = score >= 80 ? 'score-excellent' : 
                                           score >= 60 ? 'score-good' : 'score-poor';
                    
                    // Update metrics
                    updateMetric('cpu', data.cpu || 0);
                    updateMetric('memory', data.memory || 0);
                    updateMetric('disk', data.disk || 0);
                    
                    // Update values
                    document.getElementById('latency-value').textContent = (data.latency || 0) + 'ms';
                    document.getElementById('uptime-value').textContent = (data.uptime || 0) + 'm';
                    document.getElementById('processes-value').textContent = data.processes || 0;
                    
                    // Update alerts
                    const alertsDiv = document.getElementById('alerts');
                    if (data.alerts && data.alerts.length > 0) {
                        alertsDiv.innerHTML = data.alerts.slice(-10).reverse().map(alert => 
                            `<div class="alert ${getAlertClass(alert.message)}">[${alert.timestamp}] ${alert.message}</div>`
                        ).join('');
                    }
                    
                    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                })
                .catch(error => console.error('Update failed:', error));
        }
        
        function updateMetric(name, value) {
            const valueElement = document.getElementById(`${name}-value`);
            const progressElement = document.getElementById(`${name}-progress`);
            
            valueElement.textContent = value + '%';
            progressElement.style.width = value + '%';
            
            // Set color classes
            const level = value >= 85 ? 'critical' : value >= 75 ? 'warning' : 'normal';
            valueElement.className = `metric-value ${level}`;
            progressElement.className = `progress-fill ${level}`;
        }
        
        function getAlertClass(message) {
            if (message.includes('CRITICAL') || message.includes('🔴')) return 'critical';
            if (message.includes('WARNING') || message.includes('🟡')) return 'warning';
            return 'info';
        }
        
        setInterval(updateDashboard, 2000);
        updateDashboard();
    </script>
</head>
<body>
    <div class="header">
        <h1 class="title">⚡ BAMBHORIA QUANTUM MONITOR ⚡</h1>
        <p class="subtitle">Real-time Performance & System Health</p>
        <p><small>Last Update: <span id="last-update">--:--:--</span></small></p>
    </div>
    
    <div class="performance-score">
        <div class="score-title">🎯 PERFORMANCE SCORE</div>
        <div id="performance-score" class="score-value score-excellent">100</div>
    </div>
    
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-title">💻 CPU Usage</div>
            <div id="cpu-value" class="metric-value normal">0%</div>
            <div class="progress-bar">
                <div id="cpu-progress" class="progress-fill normal" style="width: 0%"></div>
            </div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">🧠 Memory Usage</div>
            <div id="memory-value" class="metric-value normal">0%</div>
            <div class="progress-bar">
                <div id="memory-progress" class="progress-fill normal" style="width: 0%"></div>
            </div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">💾 Disk Usage</div>
            <div id="disk-value" class="metric-value normal">0%</div>
            <div class="progress-bar">
                <div id="disk-progress" class="progress-fill normal" style="width: 0%"></div>
            </div>
        </div>
    </div>
    
    <div class="quantum-stats">
        <div class="stat-box">
            <div class="stat-label">Network Latency</div>
            <div id="latency-value" class="stat-value">0ms</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">System Uptime</div>
            <div id="uptime-value" class="stat-value">0m</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Active Processes</div>
            <div id="processes-value" class="stat-value">0</div>
        </div>
    </div>
    
    <div class="alerts">
        <h3>🚨 System Alerts</h3>
        <div id="alerts">
            <div class="alert info">System monitoring active...</div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(PERFORMANCE_HTML)

@app.route('/api/performance')
def get_performance():
    """Get current performance metrics"""
    try:
        # Get real-time system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        # Calculate performance score
        score = 100
        score -= max(0, (cpu_percent - 50) * 0.8)
        score -= max(0, (memory.percent - 60) * 0.6)
        score -= max(0, (disk.percent - 70) * 0.4)
        score = max(0, min(100, round(score, 1)))
        
        # Get process count
        process_count = len(psutil.pids())
        
        # Calculate uptime (mock for now)
        uptime_minutes = round(time.time() / 60, 1) % 1000  # Simple mock
        
        # Mock latency test
        latency = 0
        try:
            start_time = time.time()
            requests.get("http://localhost:5008/api/system_status", timeout=1)
            latency = round((time.time() - start_time) * 1000, 1)
        except:
            pass
        
        # Generate alerts based on thresholds
        alerts = []
        current_time = datetime.now().strftime("%H:%M:%S")
        
        if cpu_percent > 85:
            alerts.append({"timestamp": current_time, "message": f"🔴 CRITICAL: CPU usage at {cpu_percent}%"})
        elif cpu_percent > 75:
            alerts.append({"timestamp": current_time, "message": f"🟡 WARNING: CPU usage at {cpu_percent}%"})
            
        if memory.percent > 90:
            alerts.append({"timestamp": current_time, "message": f"🔴 CRITICAL: Memory usage at {memory.percent}%"})
        elif memory.percent > 80:
            alerts.append({"timestamp": current_time, "message": f"🟡 WARNING: Memory usage at {memory.percent}%"})
        
        return jsonify({
            "performance_score": score,
            "cpu": round(cpu_percent, 1),
            "memory": round(memory.percent, 1),
            "disk": round(disk.percent, 1),
            "latency": latency,
            "uptime": uptime_minutes,
            "processes": process_count,
            "alerts": alerts,
            "timestamp": datetime.now().isoformat()
        })
        
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
    print("🚀 Starting Bambhoria Performance Dashboard on http://localhost:5009")
    app.run(host='0.0.0.0', port=5009, debug=False)