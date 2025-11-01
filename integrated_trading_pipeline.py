"""
integrated_trading_pipeline.py
Bambhoria Complete Trading Pipeline v1.0
Author: Vikas Bambhoria

Pipeline Flow:
Mock/Live Feed → Signal Generator → Risk Manager + Order Brain → Dashboard

Components:
1. Mock/Live Feed: Real-time market data streaming
2. Signal Generator: AI-powered BUY/SELL/HOLD decisions
3. Risk Manager: Position limits and loss protection
4. Order Brain: Trade execution (paper/live)
5. Dashboard: Real-time monitoring and visualization
"""

import asyncio
import websockets
import json
import time
import requests
from signal_generator import process_tick
from datetime import datetime

# ---------- CONFIG ----------
FEED_SOURCE = "http://localhost:5002/api/ticks"  # Tick receiver endpoint
DASHBOARD_URL = "http://localhost:5003"          # Dashboard server
WS_FEED_URL = "ws://localhost:8765/ws"           # WebSocket feed (if available)
UPDATE_INTERVAL = 1.0                            # Seconds between feed checks

class TradingPipeline:
    def __init__(self):
        self.running = False
        self.total_signals = 0
        self.total_trades = 0
        self.total_pnl = 0.0
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    async def start_pipeline(self):
        """Main pipeline orchestrator"""
        self.running = True
        self.log("🚀 Starting Bambhoria Trading Pipeline...")
        self.log("📊 Pipeline: Feed → AI Signals → Risk Check → Orders → Dashboard")
        
        # Try WebSocket feed first, fallback to HTTP polling
        try:
            await self.connect_websocket_feed()
        except Exception as e:
            self.log(f"⚠️  WebSocket unavailable: {e}")
            self.log("🔄 Switching to HTTP polling mode...")
            await self.poll_http_feed()
    
    async def connect_websocket_feed(self):
        """Connect to WebSocket live feed"""
        self.log(f"🔌 Connecting to WebSocket: {WS_FEED_URL}")
        async with websockets.connect(WS_FEED_URL) as websocket:
            self.log("✅ WebSocket connected - Listening for live data...")
            async for message in websocket:
                if not self.running:
                    break
                await self.process_feed_data(json.loads(message))
    
    async def poll_http_feed(self):
        """Poll HTTP endpoint for market data"""
        self.log(f"📡 Polling HTTP feed: {FEED_SOURCE}")
        last_data = None
        
        while self.running:
            try:
                # Check for new data from tick receiver
                response = requests.get(f"{FEED_SOURCE.replace('/ticks', '/status')}", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('recent_ticks') and data != last_data:
                        # Process the most recent tick
                        recent_tick = data['recent_ticks'][0] if data['recent_ticks'] else None
                        if recent_tick:
                            await self.process_feed_data(recent_tick)
                            last_data = data
                        
            except requests.exceptions.RequestException:
                # If HTTP polling fails, generate synthetic data for demo
                await self.generate_synthetic_tick()
                
            await asyncio.sleep(UPDATE_INTERVAL)
    
    async def generate_synthetic_tick(self):
        """Generate synthetic market data for testing"""
        import random
        symbols = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICI"]
        base_prices = {"RELIANCE": 2550, "TCS": 3500, "INFY": 1450, "HDFC": 2600, "ICICI": 1200}
        
        symbol = random.choice(symbols)
        base = base_prices[symbol]
        price = base + random.uniform(-20, 20)
        
        tick = {
            "symbol": symbol,
            "last_price": round(price, 2),
            "volume": random.randint(10000, 50000),
            "ohlc": {
                "open": base,
                "high": price + random.uniform(0, 5),
                "low": price - random.uniform(0, 5)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        await self.process_feed_data(tick)
    
    async def process_feed_data(self, tick_data):
        """Process incoming market data through the pipeline"""
        try:
            # Step 1: Log incoming data
            symbol = tick_data.get("symbol", "UNKNOWN")
            price = tick_data.get("last_price", 0)
            self.log(f"📈 Feed: {symbol} @ ₹{price}")
            
            # Step 2: Send to Signal Generator (AI Analysis)
            self.total_signals += 1
            original_process_tick = process_tick
            
            # Wrap process_tick to capture trade results
            def wrapped_process_tick(tick):
                result = original_process_tick(tick)
                # Check if a trade was executed (simple heuristic)
                if "BUY" in str(result) or "SELL" in str(result):
                    self.total_trades += 1
                    # Extract PnL if available (would need to enhance signal_generator for this)
                return result
            
            wrapped_process_tick(tick_data)
            
            # Step 3: Send to Dashboard (if available)
            await self.update_dashboard(tick_data)
            
        except Exception as e:
            self.log(f"❌ Pipeline error: {e}")
    
    async def update_dashboard(self, tick_data):
        """Send updates to dashboard"""
        try:
            dashboard_data = {
                "type": "tick_update",
                "data": tick_data,
                "pipeline_stats": {
                    "total_signals": self.total_signals,
                    "total_trades": self.total_trades,
                    "total_pnl": self.total_pnl
                }
            }
            
            # Try to post to dashboard
            response = requests.post(
                f"{DASHBOARD_URL}/api/pipeline_update", 
                json=dashboard_data, 
                timeout=1
            )
            
        except requests.exceptions.RequestException:
            pass  # Dashboard update is optional
    
    def stop_pipeline(self):
        """Stop the trading pipeline"""
        self.running = False
        self.log("🛑 Pipeline stopped")
        self.log(f"📊 Final Stats: {self.total_signals} signals, {self.total_trades} trades")

# ---------- PIPELINE RUNNER ----------
async def main():
    pipeline = TradingPipeline()
    
    try:
        await pipeline.start_pipeline()
    except KeyboardInterrupt:
        pipeline.log("🔴 Interrupted by user")
    finally:
        pipeline.stop_pipeline()

if __name__ == "__main__":
    print("🚀 Bambhoria Complete Trading Pipeline")
    print("=" * 50)
    print("Pipeline Flow:")
    print("📊 Mock/Live Feed → Signal Generator → Risk Manager + Order Brain → Dashboard")
    print("=" * 50)
    print("Press Ctrl+C to stop")
    print()
    
    # Run the async pipeline
    asyncio.run(main())