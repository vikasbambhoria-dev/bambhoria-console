"""
dashboard_server.py
Bambhoria Trade Analytics Dashboard v1.0
Author: Vikas Bambhoria
"""

from flask import Flask, render_template_string, jsonify, request
import json, os, time
from threading import Thread
from datetime import datetime

app = Flask(__name__)

ORDERS_LOG = "logs/orders_log.json"
RISK_LOG = "logs/risk_log.json"
SIGNALS_BUFFER = []
PNL_SUMMARY = {"pnl_today": 0.0, "trades": 0, "symbols": {}}

# -------------- Helper readers --------------
def read_json_lines(path, tail=100):
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()[-tail:]
    return [json.loads(l) for l in lines if l.strip()]

def compute_pnl():
    orders = read_json_lines(ORDERS_LOG, 500)
    total = sum([o.get("pnl", 0) for o in orders])
    PNL_SUMMARY["pnl_today"] = round(total, 2)
    PNL_SUMMARY["trades"] = len(orders)
    sym_summary = {}
    for o in orders:
        s = o["symbol"]
        sym_summary.setdefault(s, 0)
        sym_summary[s] += o.get("pnl", 0)
    PNL_SUMMARY["symbols"] = sym_summary

# -------------- API routes --------------
@app.route("/")
def home():
    compute_pnl()
    return render_template_string(HTML_PAGE, pnl=PNL_SUMMARY)

@app.route("/api/orders", methods=["POST"])
def api_orders():
    data = request.json
    with open(ORDERS_LOG, "a") as f:
        f.write(json.dumps(data)+"\n")
    return jsonify({"ok": True})

@app.route("/api/signals", methods=["POST"])
def api_signals():
    data = request.json
    data["time"] = datetime.now().strftime("%H:%M:%S")
    SIGNALS_BUFFER.append(data)
    if len(SIGNALS_BUFFER) > 50:
        SIGNALS_BUFFER.pop(0)
    return jsonify({"ok": True})

@app.route("/api/pnl")
def api_pnl():
    compute_pnl()
    return jsonify(PNL_SUMMARY)

@app.route("/api/signals")
def api_get_signals():
    return jsonify(SIGNALS_BUFFER)

@app.route("/api/health")
def api_health():
    import psutil, time
    return jsonify({
        "cpu": psutil.cpu_percent(),
        "mem": psutil.virtual_memory().percent,
        "time": time.strftime("%H:%M:%S")
    })

# -------------- HTML/JS Frontend --------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Bambhoria God-Eye Dashboard</title>
<meta charset="utf-8" />
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
async function refresh(){
  let pnl = await fetch('/api/pnl').then(r=>r.json());
  let sigs = await fetch('/api/signals').then(r=>r.json());
  document.getElementById('pnl').innerText = 'PnL ₹'+pnl.pnl_today.toFixed(2)+' | Trades '+pnl.trades;
  let symbols = Object.keys(pnl.symbols);
  let values = Object.values(pnl.symbols);
  Plotly.newPlot('chart',[{x:symbols,y:values,type:'bar',marker:{color:'steelblue'}}]);
  let t = sigs.map(s=>`<tr><td>${s.time}</td><td>${s.symbol}</td><td>${s.signal||'--'}</td><td>${s.conf||'--'}</td></tr>`).join('');
  document.getElementById('sigtable').innerHTML = t;
}
setInterval(refresh,2000);
</script>
<style>
body{font-family:Arial;margin:20px;background:#f4f4f8;}
#chart{width:80%;height:400px;margin:auto;}
table{border-collapse:collapse;width:80%;margin:auto;}
th,td{border:1px solid #ccc;padding:5px;text-align:center;}
th{background:#333;color:#fff;}
</style>
</head>
<body>
<h2 align="center">🧠 Bambhoria God-Eye Analytics Dashboard v1.0</h2>
<div align="center" id="pnl" style="font-size:18px;font-weight:bold;color:#006400;">Loading...</div>
<div id="chart"></div>
<h3 align="center">📈 Recent Signals</h3>
<table><thead><tr><th>Time</th><th>Symbol</th><th>Signal</th><th>Confidence</th></tr></thead>
<tbody id="sigtable"><tr><td colspan="4">Waiting for data...</td></tr></tbody></table>
</body>
</html>
"""

# -------------- Run server --------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)