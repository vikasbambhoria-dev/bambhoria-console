"""
complete_trading_system.py
Bambhoria Complete Trading System v1.0
Author: Vikas Bambhoria

Architecture Flow:
Mock/Live Feed → Signal Generator → Order Brain → Risk Manager → Dashboard
                                         ↘
                                          ↘ Logs/PnL/Alerts → Visualization

This module orchestrates the complete trading pipeline with all components.
"""

import asyncio
import time
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import requests

# Import our core components
from signal_generator import process_tick, generate_signal, make_features
from core_engine.order_control import OrderBrain
from core_engine.risk_manager import RiskManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TradingSystem")

class CompleteTradingSystem:
    def __init__(self):
        # Initialize core components
        self.signal_generator_active = True
        self.order_brain = OrderBrain(mode="paper")
        self.risk_manager = RiskManager(max_daily_loss=5000, max_position_size=100)
        
        # Dashboard endpoints
        self.dashboard_urls = {
            "analytics": "http://localhost:5006",
            "websocket": "http://localhost:5007", 
            "pipeline": "http://localhost:5005",
            "monitor": "http://localhost:5008"  # New system monitor
        }
        
        # System statistics
        self.stats = {
            "feeds_processed": 0,
            "signals_generated": 0,
            "orders_placed": 0,
            "risk_blocks": 0,
            "total_pnl": 0.0,
            "start_time": time.time(),  # Track system start time
            "alerts": []
        }
        
        # Component status tracking
        self.component_status = {
            "feed_source": "Unknown",
            "signal_generator": "Active",
            "order_brain": "Active", 
            "risk_manager": "Active",
            "dashboard": "Unknown"
        }

    def log_alert(self, level, message):
        """Log alerts for visualization"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        self.stats["alerts"].append(alert)
        if len(self.stats["alerts"]) > 100:  # Keep last 100 alerts
            self.stats["alerts"] = self.stats["alerts"][-100:]
        
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)
        
        # Send to system monitor
        self.update_system_monitor()

    def update_system_monitor(self):
        """Send updates to system monitor dashboard"""
        try:
            monitor_data = {
                "stats": {
                    "feeds_processed": self.stats["feeds_processed"],
                    "signals_generated": self.stats["signals_generated"],
                    "orders_placed": self.stats["orders_placed"],
                    "risk_blocks": self.stats["risk_blocks"],
                    "total_pnl": self.stats["total_pnl"],
                    "uptime": time.time() - self.stats.get("start_time", time.time())
                },
                "component_status": self.component_status,
                "alerts": self.stats["alerts"]
            }
            
            requests.post(
                f"{self.dashboard_urls['monitor']}/api/system_update",
                json=monitor_data,
                timeout=1
            )
        except:
            pass  # Silent fail for monitoring

    def process_feed_data(self, tick_data):
        """
        Complete pipeline: Feed → Signal → Order → Risk → Dashboard
        """
        try:
            # Step 1: Feed Processing
            self.stats["feeds_processed"] += 1
            self.component_status["feed_source"] = "Active"
            
            symbol = tick_data.get("symbol", "UNKNOWN")
            price = tick_data.get("last_price", 0)
            
            self.log_alert("INFO", f"📡 Feed: {symbol} @ ₹{price}")
            
            # Update system monitor
            self.update_system_monitor()
            
            # Step 2: Signal Generation
            signal, confidence = generate_signal(tick_data)
            
            if confidence >= 0.65:  # Signal threshold
                self.stats["signals_generated"] += 1
                self.component_status["signal_generator"] = "Active"
                self.log_alert("INFO", f"🧠 Signal: {symbol} {signal} (conf={confidence:.2f})")
                
                # Send signal to dashboards
                self.send_to_dashboards("signals", {
                    "symbol": symbol,
                    "signal": signal,
                    "confidence": confidence,
                    "price": price,
                    "timestamp": time.time()
                })
                
                # Step 3: Order Brain Processing
                if signal in ["BUY", "SELL"]:
                    qty = 10  # Default quantity
                    
                    # Step 4: Risk Manager Check
                    risk_result = self.risk_manager.check_risk(symbol, qty, price, signal)
                    
                    if risk_result.get('allowed', False):
                        # Step 5: Execute Order
                        order_result = self.order_brain.place_order(symbol, signal, qty, price)
                        self.stats["orders_placed"] += 1
                        
                        # Update P&L
                        if hasattr(order_result, 'get') and 'pnl' in order_result:
                            self.stats["total_pnl"] += order_result['pnl']
                        
                        self.log_alert("INFO", f"✅ Order: {signal} {symbol} x{qty} @ ₹{price}")
                        
                        # Send order to dashboards
                        self.send_to_dashboards("orders", {
                            "symbol": symbol,
                            "side": signal,
                            "qty": qty,
                            "price": price,
                            "pnl": order_result.get('pnl', 0) if hasattr(order_result, 'get') else 0,
                            "mode": "paper",
                            "time": datetime.now().isoformat()
                        })
                        
                    else:
                        self.stats["risk_blocks"] += 1
                        self.log_alert("WARNING", f"🛡️ Risk Block: {symbol} {signal} - {risk_result.get('reason', 'Unknown')}")
            
            # Step 6: Update Dashboard with System Stats
            self.update_system_dashboard()
            
        except Exception as e:
            self.log_alert("ERROR", f"Pipeline error: {str(e)}")

    def send_to_dashboards(self, endpoint, data):
        """Send data to all dashboard endpoints"""
        for dashboard_name, base_url in self.dashboard_urls.items():
            try:
                url = f"{base_url}/api/{endpoint}"
                response = requests.post(url, json=data, timeout=1)
                if response.status_code == 200:
                    self.component_status["dashboard"] = "Active"
            except:
                pass  # Dashboard posting is optional

    def update_system_dashboard(self):
        """Update dashboard with complete system status"""
        system_data = {
            "stats": self.stats.copy(),
            "component_status": self.component_status.copy(),
            "uptime": time.time() - self.stats["start_time"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to pipeline dashboard
        try:
            requests.post(f"{self.dashboard_urls['pipeline']}/api/system_update", 
                         json=system_data, timeout=1)
        except:
            pass

    def generate_synthetic_feed(self):
        """Generate synthetic market data for testing"""
        import random
        
        symbols = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICI", "WIPRO", "ONGC", "ITC", "SBIN"]
        base_prices = {
            "RELIANCE": 2550, "TCS": 3500, "INFY": 1450, "HDFC": 2600, 
            "ICICI": 1200, "WIPRO": 420, "ONGC": 180, "ITC": 460, "SBIN": 520
        }
        
        symbol = random.choice(symbols)
        base = base_prices[symbol]
        
        # Generate realistic price movement
        price_change = random.uniform(-0.03, 0.03)  # ±3% movement
        current_price = base * (1 + price_change)
        
        # Generate OHLC data
        high = current_price * random.uniform(1.0, 1.02)
        low = current_price * random.uniform(0.98, 1.0)
        open_price = base * random.uniform(0.99, 1.01)
        
        tick = {
            "symbol": symbol,
            "last_price": round(current_price, 2),
            "volume": random.randint(10000, 50000),
            "ohlc": {
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(current_price, 2)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return tick

    async def run_live_system(self, duration_seconds=300):
        """Run the complete trading system"""
        self.log_alert("INFO", "🚀 Starting Complete Bambhoria Trading System")
        self.log_alert("INFO", "📊 Pipeline: Feed → Signal → Order → Risk → Dashboard")
        
        start_time = time.time()
        tick_count = 0
        
        while time.time() - start_time < duration_seconds:
            try:
                # Generate or receive market data
                tick_data = self.generate_synthetic_feed()
                
                # Process through complete pipeline
                self.process_feed_data(tick_data)
                
                tick_count += 1
                
                # Wait before next tick
                await asyncio.sleep(2)  # 2-second intervals
                
                # Log progress every 10 ticks
                if tick_count % 10 == 0:
                    elapsed = time.time() - start_time
                    self.log_alert("INFO", f"📈 Progress: {tick_count} ticks, {self.stats['signals_generated']} signals, {self.stats['orders_placed']} orders")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.log_alert("ERROR", f"System error: {str(e)}")
                await asyncio.sleep(1)
        
        # Final summary
        elapsed = time.time() - start_time
        self.log_alert("INFO", f"🏁 Trading session complete:")
        self.log_alert("INFO", f"   📊 Feeds processed: {self.stats['feeds_processed']}")
        self.log_alert("INFO", f"   🧠 Signals generated: {self.stats['signals_generated']}")
        self.log_alert("INFO", f"   ⚡ Orders placed: {self.stats['orders_placed']}")
        self.log_alert("INFO", f"   🛡️ Risk blocks: {self.stats['risk_blocks']}")
        self.log_alert("INFO", f"   💰 Total P&L: ₹{self.stats['total_pnl']:.2f}")
        self.log_alert("INFO", f"   ⏱️ Runtime: {elapsed:.1f} seconds")

# Convenience functions for direct usage
def start_trading_system(duration_minutes=5):
    """Start the complete trading system for specified duration"""
    system = CompleteTradingSystem()
    asyncio.run(system.run_live_system(duration_minutes * 60))

def demo_single_tick():
    """Demo a single tick through the complete pipeline"""
    system = CompleteTradingSystem()
    tick = system.generate_synthetic_feed()
    print(f"Generated tick: {json.dumps(tick, indent=2)}")
    system.process_feed_data(tick)
    return system.stats

if __name__ == "__main__":
    print("🚀 Bambhoria Complete Trading System")
    print("=" * 60)
    print("Architecture:")
    print("Mock/Live Feed → Signal Generator → Order Brain → Risk Manager → Dashboard")
    print("                                         ↘")
    print("                                          ↘ Logs/PnL/Alerts → Visualization")
    print("=" * 60)
    print()
    
    # Run demo
    start_trading_system(duration_minutes=2)  # 2-minute demo