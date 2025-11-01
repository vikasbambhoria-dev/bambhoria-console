"""
world_class_engine.py
Bambhoria World-Class Trading Engine v1.0
Author: Vikas Bambhoria
Purpose:
 - Real-time adaptive learning with immediate feedback
 - Multi-model AI ensemble with weighted voting
 - Advanced risk intelligence with dynamic position sizing
The best trading engine in the world 🌍
"""

import joblib, json, os, time, numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

# ---------- STEP 1: REAL-TIME SELF-LEARNING ENGINE ----------
class AdaptiveLearningEngine:
    """
    Learns from every trade in real-time and adapts strategy weights
    """
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.trade_history = deque(maxlen=1000)  # Keep last 1000 trades
        self.pattern_performance = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total_pnl': 0, 'weight': 1.0
        })
        self.strategy_weights = {}
        self.load_weights()
    
    def load_weights(self):
        """Load learned weights from disk"""
        weights_file = Path("models/adaptive_weights.json")
        if weights_file.exists():
            with open(weights_file, 'r') as f:
                data = json.load(f)
                self.pattern_performance = defaultdict(lambda: {
                    'wins': 0, 'losses': 0, 'total_pnl': 0, 'weight': 1.0
                }, data.get('patterns', {}))
                self.strategy_weights = data.get('strategy_weights', {})
            print(f"✅ Loaded {len(self.pattern_performance)} learned patterns")
    
    def save_weights(self):
        """Save learned weights to disk"""
        weights_file = Path("models/adaptive_weights.json")
        os.makedirs(weights_file.parent, exist_ok=True)
        with open(weights_file, 'w') as f:
            json.dump({
                'patterns': dict(self.pattern_performance),
                'strategy_weights': self.strategy_weights,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def record_trade(self, pattern_key: str, pnl: float, won: bool):
        """Record trade outcome and update weights immediately"""
        perf = self.pattern_performance[pattern_key]
        
        if won:
            perf['wins'] += 1
        else:
            perf['losses'] += 1
        
        perf['total_pnl'] += pnl
        
        # Adaptive weight update using reinforcement learning
        total_trades = perf['wins'] + perf['losses']
        win_rate = perf['wins'] / total_trades if total_trades > 0 else 0.5
        
        # Reward successful patterns, penalize losing patterns
        if won:
            perf['weight'] = min(2.0, perf['weight'] * (1 + self.learning_rate))
        else:
            perf['weight'] = max(0.1, perf['weight'] * (1 - self.learning_rate))
        
        # Store in history
        self.trade_history.append({
            'pattern': pattern_key,
            'pnl': pnl,
            'won': won,
            'timestamp': datetime.now().isoformat(),
            'new_weight': perf['weight']
        })
        
        # Auto-save every 10 trades
        if len(self.trade_history) % 10 == 0:
            self.save_weights()
        
        return perf['weight']
    
    def get_pattern_weight(self, pattern_key: str) -> float:
        """Get current weight for a pattern"""
        return self.pattern_performance[pattern_key]['weight']
    
    def get_top_patterns(self, n: int = 10) -> List[Tuple[str, Dict]]:
        """Get top performing patterns"""
        sorted_patterns = sorted(
            self.pattern_performance.items(),
            key=lambda x: x[1]['weight'],
            reverse=True
        )
        return sorted_patterns[:n]


# ---------- STEP 2: MULTI-MODEL AI ENSEMBLE ----------
class MultiModelEnsemble:
    """
    Combines multiple AI models with weighted voting
    """
    def __init__(self):
        self.models = {}
        self.model_weights = {
            'lightgbm': 0.4,
            'neural_insight': 0.35,
            'reinforcement': 0.25
        }
        self.model_performance = defaultdict(lambda: {
            'correct': 0, 'total': 0, 'accuracy': 0.0
        })
        self.load_models()
    
    def load_models(self):
        """Load all available AI models"""
        # Load LightGBM base model
        lgb_path = Path("models/godeye_lgbm_model.pkl")
        if lgb_path.exists():
            self.models['lightgbm'] = joblib.load(lgb_path)
            print("✅ LightGBM model loaded")
        
        # Neural Insight model (pattern-based)
        try:
            from neural_insight_engine import suggest_insight
            self.models['neural_insight'] = suggest_insight
            print("✅ Neural Insight engine loaded")
        except:
            print("⚠️  Neural Insight not available")
        
        # TODO: Load reinforcement learning model when ready
        print(f"✅ Ensemble ready with {len(self.models)} models")
    
    def predict_ensemble(self, features: pd.DataFrame, context: Dict) -> Dict:
        """
        Get weighted prediction from all models
        Returns: {'signal': 'BUY'|'SELL'|'HOLD', 'confidence': float, 'votes': dict}
        """
        votes = {}
        confidences = {}
        
        # LightGBM prediction
        if 'lightgbm' in self.models:
            model = self.models['lightgbm']
            proba = model.predict_proba(features)[0]
            signal_idx = int(np.argmax(proba))
            labels = {0: "SELL", 1: "HOLD", 2: "BUY"}
            votes['lightgbm'] = labels.get(signal_idx, "HOLD")
            confidences['lightgbm'] = float(np.max(proba))
        
        # Neural Insight prediction
        if 'neural_insight' in self.models:
            insight_func = self.models['neural_insight']
            symbol = context.get('symbol', 'UNKNOWN')
            hour = datetime.now().hour
            price = context.get('price', 0)
            
            # Get insight for BUY and SELL
            buy_insight = insight_func(symbol, 'BUY', hour, 10, price)
            sell_insight = insight_func(symbol, 'SELL', hour, 10, price)
            
            if buy_insight['prob'] > sell_insight['prob']:
                votes['neural_insight'] = 'BUY'
                confidences['neural_insight'] = buy_insight['prob']
            else:
                votes['neural_insight'] = 'SELL'
                confidences['neural_insight'] = sell_insight['prob']
        
        # Weighted voting
        weighted_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        for model_name, signal in votes.items():
            weight = self.model_weights.get(model_name, 0.33)
            confidence = confidences.get(model_name, 0.5)
            weighted_scores[signal] += weight * confidence
        
        # Final decision
        final_signal = max(weighted_scores, key=weighted_scores.get)
        final_confidence = weighted_scores[final_signal]
        
        return {
            'signal': final_signal,
            'confidence': final_confidence,
            'votes': votes,
            'confidences': confidences,
            'model_weights': self.model_weights
        }
    
    def update_model_performance(self, model_name: str, correct: bool):
        """Update model performance tracking"""
        perf = self.model_performance[model_name]
        perf['total'] += 1
        if correct:
            perf['correct'] += 1
        perf['accuracy'] = perf['correct'] / perf['total']
        
        # Adaptive weight adjustment
        if perf['total'] > 20:  # Wait for sufficient data
            # Increase weight of accurate models
            if perf['accuracy'] > 0.6:
                self.model_weights[model_name] = min(0.5, self.model_weights[model_name] * 1.05)
            else:
                self.model_weights[model_name] = max(0.1, self.model_weights[model_name] * 0.95)
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}


# ---------- STEP 3: ADVANCED RISK INTELLIGENCE ----------
class AdvancedRiskManager:
    """
    Dynamic position sizing, correlation analysis, drawdown protection
    """
    def __init__(self, initial_capital: float = 100000):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.max_drawdown = 0.0
        self.peak_capital = initial_capital
        
        # Risk parameters
        self.max_position_size_pct = 0.02  # 2% of capital per trade
        self.max_portfolio_heat = 0.10  # 10% total risk
        self.max_correlated_positions = 3
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
    
    def calculate_position_size(self, price: float, volatility: float, confidence: float) -> int:
        """
        Dynamic position sizing based on Kelly Criterion and volatility
        """
        # Base position size (2% of capital)
        base_size = (self.capital * self.max_position_size_pct) / price
        
        # Adjust for confidence (higher confidence = larger position)
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5x to 1.0x
        
        # Adjust for volatility (higher volatility = smaller position)
        volatility_multiplier = 1.0 / (1.0 + volatility * 10)
        
        # Calculate final size
        position_size = base_size * confidence_multiplier * volatility_multiplier
        
        return max(1, int(position_size))
    
    def check_risk_limits(self, symbol: str, side: str, qty: int, price: float) -> Dict:
        """
        Comprehensive risk checks before trade execution
        """
        # Check daily loss limit
        if self.daily_pnl < -(self.capital * self.daily_loss_limit):
            return {
                'allowed': False,
                'reason': f'Daily loss limit reached: ₹{self.daily_pnl:.2f}'
            }
        
        # Check portfolio heat
        current_risk = sum([abs(p.get('unrealized_pnl', 0)) for p in self.positions.values()])
        risk_pct = current_risk / self.capital
        
        if risk_pct > self.max_portfolio_heat:
            return {
                'allowed': False,
                'reason': f'Portfolio heat too high: {risk_pct*100:.1f}%'
            }
        
        # Check position count
        if len(self.positions) >= 10:
            return {
                'allowed': False,
                'reason': 'Maximum positions reached (10)'
            }
        
        # Check drawdown
        current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
        if current_drawdown > 0.20:  # 20% max drawdown
            return {
                'allowed': False,
                'reason': f'Max drawdown exceeded: {current_drawdown*100:.1f}%'
            }
        
        return {'allowed': True, 'reason': 'All risk checks passed'}
    
    def update_capital(self, pnl: float):
        """Update capital and track drawdown"""
        self.capital += pnl
        self.daily_pnl += pnl
        
        # Reset daily PnL if new day
        if datetime.now().date() != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = datetime.now().date()
        
        # Update peak and drawdown
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        
        current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        total_positions = len(self.positions)
        total_exposure = sum([p.get('qty', 0) * p.get('price', 0) for p in self.positions.values()])
        current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
        
        return {
            'capital': self.capital,
            'daily_pnl': self.daily_pnl,
            'total_positions': total_positions,
            'total_exposure': total_exposure,
            'current_drawdown': current_drawdown,
            'max_drawdown': self.max_drawdown,
            'return_pct': ((self.capital - self.initial_capital) / self.initial_capital) * 100
        }


# ---------- WORLD-CLASS ENGINE ORCHESTRATOR ----------
class WorldClassEngine:
    """
    Combines all three components into the best trading engine
    """
    def __init__(self, initial_capital: float = 100000):
        print("🌍 Initializing World-Class Trading Engine...")
        self.learner = AdaptiveLearningEngine(learning_rate=0.1)
        self.ensemble = MultiModelEnsemble()
        self.risk_manager = AdvancedRiskManager(initial_capital)
        print("✅ World-Class Engine ready!\n")
    
    def process_signal(self, tick: Dict, features: pd.DataFrame) -> Optional[Dict]:
        """
        Process a single tick through the world-class engine
        """
        symbol = tick.get('symbol', 'UNKNOWN')
        price = tick.get('last_price', 0)
        
        # Step 1: Get ensemble prediction
        context = {'symbol': symbol, 'price': price}
        prediction = self.ensemble.predict_ensemble(features, context)
        
        signal = prediction['signal']
        confidence = prediction['confidence']
        
        if signal == 'HOLD':
            return None
        
        # Step 2: Check learned pattern weight
        hour = datetime.now().hour
        pattern_key = f"{symbol}_{signal}_{hour}"
        pattern_weight = self.learner.get_pattern_weight(pattern_key)
        
        # Adjust confidence based on pattern performance
        adjusted_confidence = confidence * pattern_weight
        
        print(f"\n📊 Signal: {signal} | Confidence: {confidence:.2%} → {adjusted_confidence:.2%} (pattern weight: {pattern_weight:.2f})")
        print(f"   Votes: {prediction['votes']}")
        
        # Step 3: Calculate dynamic position size
        volatility = features.get('volatility', [0.01])[0] if len(features) > 0 else 0.01
        qty = self.risk_manager.calculate_position_size(price, volatility, adjusted_confidence)
        
        # Step 4: Risk checks
        risk_check = self.risk_manager.check_risk_limits(symbol, signal, qty, price)
        
        if not risk_check['allowed']:
            print(f"🚫 Risk check failed: {risk_check['reason']}")
            return None
        
        # Prepare trade order
        trade_order = {
            'symbol': symbol,
            'side': signal,
            'qty': qty,
            'price': price,
            'confidence': adjusted_confidence,
            'pattern_key': pattern_key,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Trade approved: {signal} {qty} {symbol} @ ₹{price}")
        
        return trade_order
    
    def record_trade_outcome(self, pattern_key: str, pnl: float):
        """Record trade outcome and update all components"""
        won = pnl > 0
        
        # Update adaptive learner
        new_weight = self.learner.record_trade(pattern_key, pnl, won)
        
        # Update risk manager
        self.risk_manager.update_capital(pnl)
        
        print(f"\n💰 Trade outcome: ₹{pnl:+.2f} ({'WIN' if won else 'LOSS'})")
        print(f"   Pattern weight updated: {new_weight:.2f}")
        print(f"   Capital: ₹{self.risk_manager.capital:,.2f}")
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        risk_metrics = self.risk_manager.get_risk_metrics()
        top_patterns = self.learner.get_top_patterns(5)
        
        return {
            'capital_metrics': risk_metrics,
            'top_patterns': [
                {
                    'pattern': p[0],
                    'weight': p[1]['weight'],
                    'wins': p[1]['wins'],
                    'losses': p[1]['losses'],
                    'pnl': p[1]['total_pnl']
                }
                for p in top_patterns
            ],
            'model_performance': dict(self.ensemble.model_performance)
        }


# ---------- DEMO ----------
def demo_world_class_engine():
    """Demo the world-class trading engine"""
    print("=" * 70)
    print("🌍 BAMBHORIA WORLD-CLASS TRADING ENGINE DEMO")
    print("=" * 70)
    
    engine = WorldClassEngine(initial_capital=100000)
    
    # Simulate 10 trades
    symbols = ['RELIANCE', 'TCS', 'HDFC', 'INFY', 'ICICI']
    
    for i in range(10):
        symbol = symbols[i % len(symbols)]
        base_price = {'RELIANCE': 2550, 'TCS': 3850, 'HDFC': 1650, 'INFY': 1450, 'ICICI': 950}[symbol]
        
        tick = {
            'symbol': symbol,
            'last_price': base_price + np.random.uniform(-20, 20),
            'volume': np.random.randint(10000, 50000)
        }
        
        features = pd.DataFrame([{
            'price_change': np.random.uniform(-0.02, 0.02),
            'volume_change': np.random.uniform(0.8, 1.2),
            'volatility': np.random.uniform(0.005, 0.03)
        }])
        
        # Process signal
        trade_order = engine.process_signal(tick, features)
        
        if trade_order:
            # Simulate trade outcome
            pnl = np.random.uniform(-100, 200)  # Random PnL for demo
            time.sleep(0.5)
            engine.record_trade_outcome(trade_order['pattern_key'], pnl)
    
    # Final report
    print("\n" + "=" * 70)
    print("📊 FINAL PERFORMANCE REPORT")
    print("=" * 70)
    
    report = engine.get_performance_report()
    
    print("\n💰 Capital Metrics:")
    for key, value in report['capital_metrics'].items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n🏆 Top 5 Patterns:")
    for p in report['top_patterns']:
        win_rate = p['wins'] / (p['wins'] + p['losses']) * 100 if (p['wins'] + p['losses']) > 0 else 0
        print(f"   {p['pattern']}: Weight={p['weight']:.2f}, WinRate={win_rate:.1f}%, PnL=₹{p['pnl']:+.2f}")
    
    print("\n✅ World-Class Engine demo complete!")


if __name__ == "__main__":
    demo_world_class_engine()
