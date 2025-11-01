"""
🚀 BAMBHORIA GOD-EYE V52 - LIVE PERFORMANCE DASHBOARD 🚀
=====================================================
Real-Time Performance Visualization Dashboard
The Ultimate Trading Interface for World-Class Applications
=====================================================
"""

import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import sqlite3
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingDatabase:
    """Database manager for trading data storage and retrieval"""
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                pnl REAL DEFAULT 0,
                strategy TEXT,
                confidence REAL,
                market_regime TEXT
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_pnl REAL NOT NULL,
                account_balance REAL NOT NULL,
                open_positions INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                signals_count INTEGER DEFAULT 0,
                trades_count INTEGER DEFAULT 0
            )
        ''')
        
        # Market data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                volume INTEGER,
                market_regime TEXT,
                sentiment_score REAL DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("📊 Database initialized successfully")
    
    def insert_trade(self, trade_data: Dict) -> None:
        """Insert a new trade record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (symbol, action, quantity, price, pnl, strategy, confidence, market_regime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data.get('symbol'),
            trade_data.get('action'),
            trade_data.get('quantity'),
            trade_data.get('price'),
            trade_data.get('pnl', 0),
            trade_data.get('strategy'),
            trade_data.get('confidence'),
            trade_data.get('market_regime')
        ))
        
        conn.commit()
        conn.close()
    
    def insert_performance_metric(self, metrics: Dict) -> None:
        """Insert performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_metrics 
            (total_pnl, account_balance, open_positions, win_rate, sharpe_ratio, max_drawdown, signals_count, trades_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.get('total_pnl', 0),
            metrics.get('account_balance', 0),
            metrics.get('open_positions', 0),
            metrics.get('win_rate', 0),
            metrics.get('sharpe_ratio', 0),
            metrics.get('max_drawdown', 0),
            metrics.get('signals_count', 0),
            metrics.get('trades_count', 0)
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_trades(self, limit: int = 100) -> List[Dict]:
        """Get recent trades"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM trades 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        trades = []
        for row in cursor.fetchall():
            trades.append({
                'id': row[0],
                'timestamp': row[1],
                'symbol': row[2],
                'action': row[3],
                'quantity': row[4],
                'price': row[5],
                'pnl': row[6],
                'strategy': row[7],
                'confidence': row[8],
                'market_regime': row[9]
            })
        
        conn.close()
        return trades
    
    def get_performance_history(self, hours: int = 24) -> List[Dict]:
        """Get performance history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_time = datetime.now() - timedelta(hours=hours)
        
        cursor.execute('''
            SELECT * FROM performance_metrics 
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        ''', (since_time,))
        
        metrics = []
        for row in cursor.fetchall():
            metrics.append({
                'timestamp': row[1],
                'total_pnl': row[2],
                'account_balance': row[3],
                'open_positions': row[4],
                'win_rate': row[5],
                'sharpe_ratio': row[6],
                'max_drawdown': row[7],
                'signals_count': row[8],
                'trades_count': row[9]
            })
        
        conn.close()
        return metrics


class LivePerformanceDashboard:
    """Advanced Real-Time Performance Dashboard"""
    
    def __init__(self, port: int = 5001):
        self.port = port
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'bambhoria_godeye_v52_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Database
        self.db = TradingDatabase()
        
        # Real-time data storage
        self.current_metrics = {
            'total_pnl': 0.0,
            'account_balance': 100000.0,
            'open_positions': 0,
            'win_rate': 0.0,
            'signals_count': 0,
            'trades_count': 0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
        self.recent_trades = []
        self.price_data = {}
        self.strategy_performance = {}
        
        # Data callbacks
        self.trade_callbacks = []
        self.metric_callbacks = []
        
        self.setup_routes()
        self.setup_socketio_events()
        
        logger.info("🚀 Live Performance Dashboard initialized")
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return self.render_dashboard_template()
        
        @self.app.route('/api/current_metrics')
        def get_current_metrics():
            """Get current performance metrics"""
            return jsonify(self.current_metrics)
        
        @self.app.route('/api/recent_trades')
        def get_recent_trades():
            """Get recent trades"""
            return jsonify(self.recent_trades[-50:])  # Last 50 trades
        
        @self.app.route('/api/performance_history')
        def get_performance_history():
            """Get performance history"""
            hours = request.args.get('hours', 24, type=int)
            history = self.db.get_performance_history(hours)
            return jsonify(history)
        
        @self.app.route('/api/strategy_performance')
        def get_strategy_performance():
            """Get strategy performance breakdown"""
            return jsonify(self.strategy_performance)
        
        @self.app.route('/api/price_data')
        def get_price_data():
            """Get real-time price data"""
            return jsonify(self.price_data)
    
    def setup_socketio_events(self):
        """Setup SocketIO events for real-time updates"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info("📱 Client connected to dashboard")
            # Send initial data
            emit('metrics_update', self.current_metrics)
            emit('trades_update', self.recent_trades[-10:])
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info("📱 Client disconnected from dashboard")
    
    def render_dashboard_template(self) -> str:
        """Render the dashboard HTML template"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Bambhoria God-Eye V52 - Live Dashboard</title>
    <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Arial', sans-serif; 
            background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
            color: #ffffff;
            overflow-x: hidden;
        }
        .header {
            background: linear-gradient(90deg, #ff6b35 0%, #f7931e 50%, #ff6b35 100%);
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(255, 107, 53, 0.3);
        }
        .header h1 {
            font-size: 2.5em;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .header p {
            font-size: 1.2em;
            margin-top: 10px;
            opacity: 0.9;
        }
        .dashboard-container {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .card {
            background: rgba(26, 31, 46, 0.8);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 107, 53, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
        }
        .card h3 {
            color: #ff6b35;
            margin-bottom: 15px;
            font-size: 1.3em;
            text-align: center;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .metric {
            text-align: center;
            padding: 15px;
            background: rgba(255, 107, 53, 0.1);
            border-radius: 10px;
            border: 1px solid rgba(255, 107, 53, 0.3);
        }
        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #00ff88;
            margin-bottom: 5px;
        }
        .metric-label {
            font-size: 0.9em;
            color: #cccccc;
        }
        .trades-list {
            max-height: 300px;
            overflow-y: auto;
            margin-top: 10px;
        }
        .trade-item {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            margin: 5px 0;
            background: rgba(255, 107, 53, 0.1);
            border-radius: 8px;
            border-left: 4px solid #00ff88;
            font-size: 0.9em;
        }
        .trade-buy { border-left-color: #00ff88; }
        .trade-sell { border-left-color: #ff4757; }
        .chart-container {
            grid-column: span 2;
            height: 300px;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #00ff88;
            margin-right: 10px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .performance-summary {
            grid-column: span 3;
            text-align: center;
            background: linear-gradient(135deg, rgba(255, 107, 53, 0.2), rgba(247, 147, 30, 0.2));
        }
        .strategy-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .strategy-card {
            background: rgba(0, 255, 136, 0.1);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(0, 255, 136, 0.3);
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 BAMBHORIA GOD-EYE V52 🚀</h1>
        <p><span class="status-indicator"></span>LIVE PERFORMANCE DASHBOARD - WORLD CLASS TRADING SYSTEM</p>
    </div>
    
    <div class="dashboard-container">
        <!-- Performance Metrics -->
        <div class="card">
            <h3>📊 Live Performance Metrics</h3>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value" id="total-pnl">₹0</div>
                    <div class="metric-label">Total P&L</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="win-rate">0%</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="account-balance">₹100,000</div>
                    <div class="metric-label">Account Balance</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="trades-count">0</div>
                    <div class="metric-label">Trades Today</div>
                </div>
            </div>
        </div>
        
        <!-- Recent Trades -->
        <div class="card">
            <h3>📈 Recent Trades</h3>
            <div class="trades-list" id="trades-list">
                <div class="trade-item">
                    <span>System Starting...</span>
                    <span>Waiting for trades</span>
                </div>
            </div>
        </div>
        
        <!-- Real-Time Chart -->
        <div class="card chart-container">
            <h3>💹 P&L Performance Chart</h3>
            <canvas id="pnl-chart"></canvas>
        </div>
        
        <!-- Strategy Performance -->
        <div class="card performance-summary">
            <h3>🤖 AI Strategy Performance</h3>
            <div class="strategy-grid" id="strategy-grid">
                <div class="strategy-card">
                    <h4>SMA Cross</h4>
                    <div class="metric-value">0</div>
                    <div class="metric-label">Trades</div>
                </div>
                <div class="strategy-card">
                    <h4>RSI Mean Reversion</h4>
                    <div class="metric-value">0</div>
                    <div class="metric-label">Trades</div>
                </div>
                <div class="strategy-card">
                    <h4>Momentum</h4>
                    <div class="metric-value">0</div>
                    <div class="metric-label">Trades</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Initialize Socket.IO
        const socket = io();
        
        // Chart initialization
        const ctx = document.getElementById('pnl-chart').getContext('2d');
        const pnlChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'P&L Performance',
                    data: [],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { 
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#ffffff' }
                    },
                    y: { 
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#ffffff' }
                    }
                }
            }
        });
        
        // Socket event handlers
        socket.on('metrics_update', function(data) {
            updateMetrics(data);
            updateChart(data);
        });
        
        socket.on('trades_update', function(trades) {
            updateTradesList(trades);
        });
        
        function updateMetrics(metrics) {
            document.getElementById('total-pnl').textContent = `₹${metrics.total_pnl.toLocaleString()}`;
            document.getElementById('win-rate').textContent = `${(metrics.win_rate * 100).toFixed(1)}%`;
            document.getElementById('account-balance').textContent = `₹${metrics.account_balance.toLocaleString()}`;
            document.getElementById('trades-count').textContent = metrics.trades_count;
        }
        
        function updateTradesList(trades) {
            const tradesList = document.getElementById('trades-list');
            tradesList.innerHTML = '';
            
            trades.slice(-10).reverse().forEach(trade => {
                const tradeElement = document.createElement('div');
                tradeElement.className = `trade-item trade-${trade.action.toLowerCase()}`;
                tradeElement.innerHTML = `
                    <span>${trade.symbol} ${trade.action}</span>
                    <span>₹${trade.price.toFixed(2)}</span>
                `;
                tradesList.appendChild(tradeElement);
            });
        }
        
        function updateChart(metrics) {
            const now = new Date().toLocaleTimeString();
            
            if (pnlChart.data.labels.length > 20) {
                pnlChart.data.labels.shift();
                pnlChart.data.datasets[0].data.shift();
            }
            
            pnlChart.data.labels.push(now);
            pnlChart.data.datasets[0].data.push(metrics.total_pnl);
            pnlChart.update('none');
        }
        
        // Fetch initial data
        fetch('/api/current_metrics')
            .then(response => response.json())
            .then(data => updateMetrics(data));
        
        console.log('🚀 Bambhoria God-Eye V52 Dashboard Loaded!');
    </script>
</body>
</html>
        '''
    
    def update_metrics(self, metrics: Dict) -> None:
        """Update current metrics"""
        self.current_metrics.update(metrics)
        
        # Store in database
        self.db.insert_performance_metric(metrics)
        
        # Emit real-time update
        self.socketio.emit('metrics_update', self.current_metrics)
        
        # Call registered callbacks
        for callback in self.metric_callbacks:
            callback(metrics)
    
    def add_trade(self, trade: Dict) -> None:
        """Add a new trade"""
        trade['timestamp'] = datetime.now().isoformat()
        self.recent_trades.append(trade)
        
        # Keep only last 100 trades in memory
        if len(self.recent_trades) > 100:
            self.recent_trades = self.recent_trades[-100:]
        
        # Store in database
        self.db.insert_trade(trade)
        
        # Emit real-time update
        self.socketio.emit('trades_update', self.recent_trades[-10:])
        
        # Call registered callbacks
        for callback in self.trade_callbacks:
            callback(trade)
    
    def update_strategy_performance(self, strategy_data: Dict) -> None:
        """Update strategy performance data"""
        self.strategy_performance.update(strategy_data)
        self.socketio.emit('strategy_update', self.strategy_performance)
    
    def update_price_data(self, symbol: str, price: float, volume: int = 0) -> None:
        """Update real-time price data"""
        if symbol not in self.price_data:
            self.price_data[symbol] = []
        
        price_point = {
            'timestamp': datetime.now().isoformat(),
            'price': price,
            'volume': volume
        }
        
        self.price_data[symbol].append(price_point)
        
        # Keep only last 100 data points per symbol
        if len(self.price_data[symbol]) > 100:
            self.price_data[symbol] = self.price_data[symbol][-100:]
        
        self.socketio.emit('price_update', {symbol: price_point})
    
    def register_trade_callback(self, callback) -> None:
        """Register trade callback"""
        self.trade_callbacks.append(callback)
    
    def register_metric_callback(self, callback) -> None:
        """Register metrics callback"""
        self.metric_callbacks.append(callback)
    
    def start_dashboard(self) -> None:
        """Start the dashboard server"""
        logger.info(f"🚀 Starting Live Performance Dashboard on port {self.port}")
        logger.info(f"🌐 Dashboard URL: http://localhost:{self.port}")
        
        # Start in a separate thread to avoid blocking
        dashboard_thread = threading.Thread(
            target=lambda: self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=False),
            daemon=True
        )
        dashboard_thread.start()
        
        return dashboard_thread
    
    def stop_dashboard(self) -> None:
        """Stop the dashboard server"""
        logger.info("🛑 Stopping Live Performance Dashboard")


def main():
    """Demo of the Live Performance Dashboard"""
    print("\n" + "="*80)
    print("🚀 BAMBHORIA GOD-EYE V52 - LIVE PERFORMANCE DASHBOARD DEMO 🚀")
    print("="*80 + "\n")
    
    # Initialize dashboard
    dashboard = LivePerformanceDashboard(port=5001)
    
    # Start dashboard
    dashboard.start_dashboard()
    
    print("🌐 Dashboard started at: http://localhost:5001")
    print("📊 Real-time performance visualization active!")
    print("\n🎯 Demo Features:")
    print("   ✅ Live P&L tracking")
    print("   ✅ Real-time trade feed")
    print("   ✅ Interactive charts")
    print("   ✅ Strategy performance")
    print("   ✅ Professional UI")
    
    # Simulate some demo data
    print("\n📈 Generating demo data...")
    
    import random
    symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICI']
    total_pnl = 0
    trades_count = 0
    
    for i in range(20):
        # Simulate trade
        symbol = random.choice(symbols)
        action = random.choice(['BUY', 'SELL'])
        price = random.uniform(1000, 4000)
        pnl = random.uniform(-500, 1500)
        total_pnl += pnl
        trades_count += 1
        
        trade = {
            'symbol': symbol,
            'action': action,
            'quantity': random.randint(10, 100),
            'price': price,
            'pnl': pnl,
            'strategy': random.choice(['SMA_Cross', 'RSI_MeanReversion', 'Momentum']),
            'confidence': random.uniform(0.6, 0.95),
            'market_regime': 'high_volatility'
        }
        
        dashboard.add_trade(trade)
        
        # Update metrics
        metrics = {
            'total_pnl': total_pnl,
            'account_balance': 100000 + total_pnl,
            'open_positions': random.randint(0, 5),
            'win_rate': random.uniform(0.7, 0.9),
            'signals_count': trades_count + random.randint(0, 10),
            'trades_count': trades_count,
            'sharpe_ratio': random.uniform(1.2, 2.5),
            'max_drawdown': random.uniform(0.02, 0.08)
        }
        
        dashboard.update_metrics(metrics)
        
        time.sleep(2)  # 2 second intervals
    
    print(f"\n✅ Demo completed! Generated {trades_count} trades")
    print(f"💰 Final P&L: ₹{total_pnl:,.0f}")
    print("\n🌐 Dashboard running at: http://localhost:5001")
    print("Press Ctrl+C to stop...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        dashboard.stop_dashboard()
        print("\n👋 Dashboard stopped")


if __name__ == "__main__":
    main()