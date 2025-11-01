"""
adaptive_signal_engine.py
Bambhoria Adaptive Signal Engine v1.0
Author: Vikas Bambhoria
Purpose:
 - Integrates Neural Insight Engine with Signal Generation
 - Creates feedback loop: Trade → Log → Pattern Learning → Signal Weight Update
 - Self-improving trading system that learns from every trade
"""

import joblib, os, time, numpy as np, pandas as pd, json
from datetime import datetime
from core_engine.order_control import OrderBrain
from core_engine.risk_manager import RiskManager
from neural_insight_engine import suggest_insight, train_pattern_model, load_trades

# ---------- CONFIG ----------
MODEL_PATH = "models/godeye_lgbm_model.pkl"
NEURAL_WEIGHTS_PATH = "models/neural_signal_weights.json"
FEATURE_ORDER = ["price_change","volume_change","volatility"]
EXECUTION_MODE = "paper"
MIN_CONFIDENCE = 0.65
NEURAL_BOOST = 0.15  # how much neural insights boost/reduce confidence
COOLDOWN_SECONDS = 10
RETRAIN_INTERVAL = 50  # retrain after N trades

# Initialize components
brain = OrderBrain(mode=EXECUTION_MODE)
risk = RiskManager()
last_trade_time = {}
trade_count = 0

# Load base ML model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print(f"✅ Base Model loaded: {MODEL_PATH}")

# Load or initialize neural weights
def load_neural_weights():
    if os.path.exists(NEURAL_WEIGHTS_PATH):
        with open(NEURAL_WEIGHTS_PATH, 'r') as f:
            weights = json.load(f)
        print(f"✅ Neural weights loaded: {len(weights)} patterns")
        return weights
    return {}

def save_neural_weights(weights):
    with open(NEURAL_WEIGHTS_PATH, 'w') as f:
        json.dump(weights, f, indent=2)

neural_weights = load_neural_weights()

# ---------- ENHANCED FEATURE BUILDER ----------
def make_features(tick):
    lp = tick["last_price"]
    ohlc = tick.get("ohlc", {})
    vol = tick.get("volume", 0)
    base = ohlc.get("open", lp)
    
    return pd.DataFrame([{
        "price_change": (lp-base)/base,
        "volume_change": vol/10000,
        "volatility": abs(ohlc.get("high",lp)-ohlc.get("low",lp))/base
    }], columns=FEATURE_ORDER)

# ---------- NEURAL-ENHANCED SIGNAL GENERATION ----------
def generate_adaptive_signal(tick):
    """Enhanced signal generation with neural insight integration"""
    global trade_count
    
    # Base ML model prediction
    X = make_features(tick)
    prob = model.predict_proba(X)[0] if hasattr(model,"predict_proba") else [0,0,0]
    signal_idx = int(np.argmax(prob))
    base_conf = float(np.max(prob))
    labels = {0:"SELL",1:"HOLD",2:"BUY"} if len(prob)==3 else {0:"SELL",1:"BUY"}
    base_signal = labels.get(signal_idx,"HOLD")
    
    # Neural insight enhancement
    symbol = tick["symbol"]
    hour = datetime.now().hour
    price = tick["last_price"]
    
    # Get neural insight for this trade setup
    neural_insight = suggest_insight(symbol, base_signal, hour, 10, price)
    neural_prob = neural_insight["prob"]
    
    # Adaptive confidence adjustment based on neural patterns
    if neural_prob > 0.7:  # High confidence neural pattern
        adjusted_conf = min(0.95, base_conf + NEURAL_BOOST)
        boost_reason = "🧠 Neural boost: Strong pattern"
    elif neural_prob < 0.4:  # Risky neural pattern  
        adjusted_conf = max(0.3, base_conf - NEURAL_BOOST)
        boost_reason = "⚠️ Neural warning: Risky pattern"
    else:
        adjusted_conf = base_conf
        boost_reason = "🤖 Neutral neural signal"
    
    # Update neural weights based on pattern
    pattern_key = f"{symbol}_{base_signal}_{hour}"
    if pattern_key not in neural_weights:
        neural_weights[pattern_key] = {"count": 0, "success_rate": 0.5, "total_pnl": 0}
    
    return {
        "signal": base_signal,
        "base_confidence": base_conf,
        "neural_confidence": neural_prob,
        "final_confidence": adjusted_conf,
        "boost_reason": boost_reason,
        "pattern_key": pattern_key
    }

# ---------- FEEDBACK LOOP PROCESSOR ----------
def update_neural_patterns():
    """Retrain neural model and update pattern weights"""
    global neural_weights
    
    print("🔄 Updating neural patterns from recent trades...")
    
    # Retrain neural insight model
    train_pattern_model()
    
    # Analyze recent trades for pattern updates
    df = load_trades()
    if not df.empty and len(df) > 5:
        # Update pattern success rates
        for pattern in neural_weights.keys():
            symbol, signal, hour = pattern.split('_')
            
            # Filter trades for this pattern
            pattern_trades = df[
                (df['symbol'] == symbol) & 
                (df['side'] == signal) & 
                (df['hour'] == int(hour))
            ]
            
            if not pattern_trades.empty:
                success_rate = (pattern_trades['pnl'] > 0).mean()
                total_pnl = pattern_trades['pnl'].sum()
                
                neural_weights[pattern]['success_rate'] = success_rate
                neural_weights[pattern]['total_pnl'] = total_pnl
                neural_weights[pattern]['count'] = len(pattern_trades)
        
        # Save updated weights
        save_neural_weights(neural_weights)
        print(f"✅ Updated {len(neural_weights)} neural patterns")

# ---------- ENHANCED EXECUTION ----------
def process_adaptive_tick(tick):
    """Process tick with neural enhancement and feedback loop"""
    global trade_count
    
    sym = tick["symbol"]
    signal_data = generate_adaptive_signal(tick)
    
    signal = signal_data["signal"]
    final_conf = signal_data["final_confidence"]
    pattern_key = signal_data["pattern_key"]
    
    now = time.time()
    
    # Enhanced filtering with neural insights
    if final_conf < MIN_CONFIDENCE or signal == "HOLD":
        print(f"⏸️ {sym}: {signal} (conf={final_conf:.2f}) - Below threshold")
        return
    
    # Cooldown check
    if sym in last_trade_time and now - last_trade_time[sym] < COOLDOWN_SECONDS:
        return
    
    last_trade_time[sym] = now
    trade_count += 1
    
    # Execute trade
    side = signal
    qty = 10
    price = tick["last_price"]
    
    print(f"🚀 ADAPTIVE SIGNAL: {sym} {signal}")
    print(f"   📊 Base Confidence: {signal_data['base_confidence']:.2f}")
    print(f"   🧠 Neural Confidence: {signal_data['neural_confidence']:.2f}")
    print(f"   ⚡ Final Confidence: {final_conf:.2f}")
    print(f"   💡 {signal_data['boost_reason']}")
    
    # Send enhanced signal to dashboard
    try:
        import requests
        enhanced_signal_data = {
            "symbol": sym,
            "signal": signal,
            "base_confidence": signal_data['base_confidence'],
            "neural_confidence": signal_data['neural_confidence'],
            "final_confidence": final_conf,
            "boost_reason": signal_data['boost_reason'],
            "price": price,
            "quantity": qty,
            "timestamp": time.time(),
            "pattern_key": pattern_key
        }
        requests.post("http://localhost:5006/api/adaptive_signals", json=enhanced_signal_data, timeout=1)
    except:
        pass  # Dashboard posting is optional
    
    # Execute order through OrderBrain
    side = signal
    qty = 10
    price = tick["last_price"]
    
    # Check risk before executing
    risk_check = risk.check_risk(sym, qty, price, side)
    if not risk_check['allowed']:
        print(f"🚫 Risk check failed: {risk_check['reason']}")
        return
    
    trade_result = brain.place_order(sym, side, qty, price)
    
    # Trigger neural retraining after sufficient trades
    if trade_count % RETRAIN_INTERVAL == 0:
        update_neural_patterns()
    
    return trade_result

# ---------- ADAPTIVE STREAM INTERFACE ----------
def run_adaptive_simulator(feed_source):
    """
    Run adaptive signal engine with neural feedback loop
    feed_source: iterable yielding tick dicts
    """
    print("🧬 Starting Adaptive Signal Engine with Neural Feedback Loop...")
    print(f"📊 Base Model: {MODEL_PATH}")
    print(f"🧠 Neural Patterns: {len(neural_weights)} loaded")
    print(f"⚙️ Mode: {EXECUTION_MODE.upper()}")
    print("─" * 60)
    
    for tick in feed_source:
        try:
            process_adaptive_tick(tick)
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n🛑 Adaptive engine stopped by user")
            break
        except Exception as e:
            print(f"❌ Error processing tick: {e}")
            continue
    
    # Final neural pattern update
    print("\n🔄 Final neural pattern update...")
    update_neural_patterns()
    print("✅ Adaptive Signal Engine completed")

# ---------- PATTERN ANALYSIS ----------
def analyze_neural_patterns():
    """Analyze current neural pattern performance"""
    print("🧠 Neural Pattern Analysis:")
    print("─" * 50)
    
    if not neural_weights:
        print("No neural patterns learned yet.")
        return
    
    # Sort patterns by success rate
    sorted_patterns = sorted(
        neural_weights.items(), 
        key=lambda x: x[1]['success_rate'], 
        reverse=True
    )
    
    for pattern, data in sorted_patterns[:10]:  # Top 10 patterns
        symbol, signal, hour = pattern.split('_')
        print(f"📈 {symbol} {signal} @{hour}h: "
              f"{data['success_rate']:.1%} success "
              f"({data['count']} trades, ₹{data['total_pnl']:+.1f} PnL)")

# ---------- DEMO ----------
if __name__ == "__main__":
    print("🧬 Bambhoria Adaptive Signal Engine v1.0")
    print("Integration: Trade → Log → Neural → Pattern → Weight Update")
    print("")
    
    # Analyze current patterns
    analyze_neural_patterns()
    print("")
    
    # Demo with enhanced fake ticks
    import random
    symbols = ["RELIANCE", "TCS", "HDFC", "INFY", "ICICI"]
    
    enhanced_ticks = []
    for i in range(30):
        sym = random.choice(symbols)
        base_price = {"RELIANCE": 2550, "TCS": 3850, "HDFC": 1650, "INFY": 1450, "ICICI": 950}[sym]
        
        enhanced_ticks.append({
            "symbol": sym,
            "last_price": base_price + random.uniform(-20, 20),
            "volume": random.randint(5000, 50000),
            "ohlc": {
                "open": base_price,
                "high": base_price + random.uniform(5, 25),
                "low": base_price - random.uniform(5, 25)
            }
        })
    
    run_adaptive_simulator(enhanced_ticks)