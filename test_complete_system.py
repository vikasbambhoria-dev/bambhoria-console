"""
test_complete_system.py
Test the complete trading system by feeding it mock data
"""

import time
import random
from complete_trading_system import CompleteTradingSystem

def generate_mock_tick(symbol):
    """Generate mock tick data"""
    base_prices = {
        "RELIANCE": 2500,
        "TCS": 3200,
        "INFY": 1800,
        "HDFCBANK": 1600,
        "ICICIBANK": 900
    }
    
    base_price = base_prices.get(symbol, 1000)
    price_change = random.uniform(-0.02, 0.02)  # ±2% change
    new_price = base_price * (1 + price_change)
    
    return {
        "symbol": symbol,
        "last_price": round(new_price, 2),
        "bid": round(new_price - 0.5, 2),
        "ask": round(new_price + 0.5, 2),
        "volume": random.randint(1000, 10000),
        "timestamp": time.time()
    }

def main():
    print("🧪 Testing Complete Trading System")
    print("=" * 40)
    
    # Initialize the trading system
    trading_system = CompleteTradingSystem()
    
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    
    print(f"📡 Starting to feed mock data for {len(symbols)} symbols...")
    print("⚡ Architecture: Feed → Signal → Order → Risk → Dashboard")
    print("-" * 40)
    
    try:
        for i in range(50):  # Generate 50 ticks
            symbol = random.choice(symbols)
            tick_data = generate_mock_tick(symbol)
            
            print(f"📊 Tick {i+1:2d}: {symbol:10s} @ ₹{tick_data['last_price']:8.2f}")
            
            # Feed data through the complete pipeline
            trading_system.process_feed_data(tick_data)
            
            time.sleep(1)  # 1-second intervals
            
    except KeyboardInterrupt:
        print("\n⏹️  Test stopped by user")
    
    # Show final stats
    print("\n📈 Final System Statistics:")
    print("-" * 30)
    stats = trading_system.stats
    print(f"Feeds Processed:     {stats['feeds_processed']:4d}")
    print(f"Signals Generated:   {stats['signals_generated']:4d}")
    print(f"Orders Placed:       {stats['orders_placed']:4d}")
    print(f"Risk Blocks:         {stats['risk_blocks']:4d}")
    print(f"Total P&L:           ₹{stats['total_pnl']:8.2f}")
    
    print(f"\n🎯 Test completed! Check the dashboards:")
    print("   • System Monitor:     http://localhost:5008")
    print("   • Analytics Dashboard: http://localhost:5006")

if __name__ == "__main__":
    main()