"""
HTTP Mock Live Feed
Sends tick data to Flask API server via HTTP POST
"""

import requests
import json
import time
import random
from datetime import datetime

class HTTPMockFeed:
    def __init__(self, api_url="http://localhost:5002/api/ticks"):
        self.api_url = api_url
        self.symbols = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICI", "WIPRO", "ONGC", "ITC", "SBIN"]
        self.base_prices = {
            "RELIANCE": 2800.0,
            "TCS": 3500.0,
            "INFY": 1450.0,
            "HDFC": 2600.0,
            "ICICI": 1200.0,
            "WIPRO": 420.0,
            "ONGC": 180.0,
            "ITC": 460.0,
            "SBIN": 520.0
        }
        self.current_prices = self.base_prices.copy()
        
    def generate_tick(self, symbol):
        """Generate realistic tick data"""
        base_price = self.base_prices[symbol]
        current_price = self.current_prices[symbol]
        
        # Random price movement (±2%)
        change_percent = random.uniform(-2.0, 2.0)
        new_price = current_price * (1 + change_percent / 100)
        
        # Keep price within ±20% of base
        min_price = base_price * 0.8
        max_price = base_price * 1.2
        new_price = max(min_price, min(max_price, new_price))
        
        self.current_prices[symbol] = new_price
        
        # Calculate change from base
        change = new_price - base_price
        change_percent_from_base = (change / base_price) * 100
        
        tick = {
            "symbol": symbol,
            "ltp": round(new_price, 2),
            "price": round(new_price, 2),
            "change": round(change, 2),
            "change_percent": round(change_percent_from_base, 2),
            "volume": random.randint(100, 1000),
            "timestamp": datetime.now().isoformat(),
            "bid": round(new_price - 0.5, 2),
            "ask": round(new_price + 0.5, 2),
            "high": round(new_price * 1.02, 2),
            "low": round(new_price * 0.98, 2)
        }
        
        return tick
        
    def send_tick(self, tick):
        """Send tick to API server"""
        try:
            response = requests.post(
                self.api_url,
                json=tick,
                timeout=2
            )
            
            if response.status_code == 200:
                direction = "📈" if tick['change_percent'] >= 0 else "📉"
                print(f"✅ Sent {tick['symbol']}: ₹{tick['ltp']} ({tick['change_percent']:+.2f}%) {direction}")
                return True
            else:
                print(f"❌ Failed to send {tick['symbol']}: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
            return False
            
    def start_streaming(self, duration_seconds=60):
        """Start streaming tick data"""
        print("🚀 Starting HTTP Mock Live Feed...")
        print(f"📡 Sending to: {self.api_url}")
        print(f"⏱️  Duration: {duration_seconds} seconds")
        print(f"📊 Symbols: {', '.join(self.symbols)}")
        print("=" * 60)
        
        start_time = time.time()
        tick_count = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                # Send ticks for all symbols
                for symbol in self.symbols:
                    tick = self.generate_tick(symbol)
                    if self.send_tick(tick):
                        tick_count += 1
                    
                    # Small delay between symbols
                    time.sleep(0.1)
                
                # Delay between rounds
                time.sleep(random.uniform(0.5, 1.5))
                
        except KeyboardInterrupt:
            print("\n🛑 Streaming stopped by user")
            
        print("=" * 60)
        print(f"📊 Total ticks sent: {tick_count}")
        print(f"⏱️  Duration: {time.time() - start_time:.1f} seconds")
        print("✅ HTTP Mock Feed completed")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HTTP Mock Live Feed")
    parser.add_argument("--url", default="http://localhost:5002/api/ticks", 
                       help="API endpoint URL")
    parser.add_argument("--duration", type=int, default=60,
                       help="Streaming duration in seconds")
    parser.add_argument("--test", action="store_true",
                       help="Send a single test tick")
    
    args = parser.parse_args()
    
    feed = HTTPMockFeed(api_url=args.url)
    
    if args.test:
        # Send single test tick
        print("🧪 Sending test tick...")
        tick = feed.generate_tick("RELIANCE")
        success = feed.send_tick(tick)
        print(f"✅ Test {'passed' if success else 'failed'}")
    else:
        # Start streaming
        feed.start_streaming(duration_seconds=args.duration)

if __name__ == "__main__":
    main()