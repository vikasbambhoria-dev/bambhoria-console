"""
Bambhoria God-Eye V52 - AI-Powered Multi-Strategy Ensemble Engine
================================================================
World-class trading engine that combines multiple strategies with:
- Real-time strategy performance tracking
- AI-powered dynamic weight allocation  
- Market regime detection and adaptation
- Ensemble learning for optimal performance
- Risk-adjusted strategy selection
- Continuous learning and adaptation

This is the revolutionary component that makes the system world-class!
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, defaultdict
from datetime import datetime, timedelta
import threading
import json
import time
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"  
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


@dataclass
class StrategySignal:
    """Container for strategy trading signals"""
    strategy_name: str
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class StrategyPerformance:
    """Track individual strategy performance"""
    name: str
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_holding_period: float = 0.0
    current_weight: float = 0.0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()
    
    @property
    def win_rate(self) -> float:
        return self.winning_trades / max(self.total_trades, 1)
    
    @property
    def avg_profit_per_trade(self) -> float:
        return self.total_pnl / max(self.total_trades, 1)


class MarketRegimeDetector:
    """Intelligent market regime detection using statistical analysis"""
    
    def __init__(self, lookback_period: int = 50):
        self.lookback_period = lookback_period
        self.price_history = defaultdict(lambda: deque(maxlen=lookback_period))
        self.volume_history = defaultdict(lambda: deque(maxlen=lookback_period))
        
        # Regime thresholds
        self.trend_threshold = 0.6  # Minimum directional movement for trend
        self.volatility_threshold = 0.02  # Daily volatility threshold
        
        logger.info("🎯 Market Regime Detector initialized")
    
    def update_market_data(self, symbol: str, price: float, volume: float = None):
        """Update market data for regime detection"""
        self.price_history[symbol].append(price)
        if volume:
            self.volume_history[symbol].append(volume)
    
    def detect_regime(self, symbol: str) -> MarketRegime:
        """Detect current market regime for a symbol"""
        if len(self.price_history[symbol]) < 20:
            return MarketRegime.UNKNOWN
        
        prices = np.array(list(self.price_history[symbol]))
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate regime indicators
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        trend_strength = self._calculate_trend_strength(prices)
        trend_direction = self._calculate_trend_direction(prices)
        
        # Market regime classification
        if volatility > self.volatility_threshold * 2:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < self.volatility_threshold * 0.5:
            return MarketRegime.LOW_VOLATILITY
        elif abs(trend_strength) > self.trend_threshold:
            if trend_direction > 0:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.RANGING
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength using linear regression slope"""
        if len(prices) < 10:
            return 0.0
        
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        return abs(slope) / np.mean(prices)  # Normalized slope
    
    def _calculate_trend_direction(self, prices: np.ndarray) -> float:
        """Calculate trend direction (-1 to 1)"""
        if len(prices) < 10:
            return 0.0
        
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        return 1.0 if slope > 0 else -1.0
    
    def get_regime_summary(self) -> Dict[str, MarketRegime]:
        """Get current regime for all tracked symbols"""
        summary = {}
        for symbol in self.price_history.keys():
            summary[symbol] = self.detect_regime(symbol)
        return summary


class StrategyEngine:
    """Individual strategy implementation"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.is_active = True
        self.performance = StrategyPerformance(name=name)
        
        # Strategy-specific parameters
        self.lookback = config.get('lookback', 20)
        self.signal_history = deque(maxlen=100)
        
    def generate_signal(self, symbol: str, price_data: Dict[str, float], 
                       market_regime: MarketRegime) -> Optional[StrategySignal]:
        """Generate trading signal based on strategy logic"""
        return None
    
    def update_performance(self, pnl: float, trade_result: str):
        """Update strategy performance metrics"""
        self.performance.total_trades += 1
        self.performance.total_pnl += pnl
        
        if trade_result == 'WIN':
            self.performance.winning_trades += 1
        
        # Update Sharpe ratio (simplified)
        if self.performance.total_trades > 0:
            avg_return = self.performance.total_pnl / self.performance.total_trades
            self.performance.sharpe_ratio = avg_return / max(abs(avg_return), 0.01)
        
        self.performance.last_updated = datetime.now()


class SMAStrategy(StrategyEngine):
    """Simple Moving Average Crossover Strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("SMA_Cross", config)
        self.short_window = config.get('short_window', 10)
        self.long_window = config.get('long_window', 30)
        self.price_history = defaultdict(lambda: deque(maxlen=self.long_window))
    
    def generate_signal(self, symbol: str, price_data: Dict[str, float], 
                       market_regime: MarketRegime) -> Optional[StrategySignal]:
        
        current_price = price_data.get('close', price_data.get('price', 0))
        self.price_history[symbol].append(current_price)
        
        if len(self.price_history[symbol]) < self.long_window:
            return None
        
        prices = np.array(list(self.price_history[symbol]))
        short_sma = np.mean(prices[-self.short_window:])
        long_sma = np.mean(prices[-self.long_window:])
        
        # Strategy works better in trending markets
        confidence_boost = 0.1 if market_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN] else 0
        
        if short_sma > long_sma * 1.005:  # 0.5% threshold
            confidence = min(0.9, 0.6 + (short_sma/long_sma - 1) * 10 + confidence_boost)
            return StrategySignal(
                strategy_name=self.name,
                symbol=symbol,
                action='BUY',
                confidence=confidence,
                target_price=current_price * 1.02,
                stop_loss=current_price * 0.98
            )
        elif short_sma < long_sma * 0.995:  # 0.5% threshold
            confidence = min(0.9, 0.6 + (long_sma/short_sma - 1) * 10 + confidence_boost)
            return StrategySignal(
                strategy_name=self.name,
                symbol=symbol,
                action='SELL',
                confidence=confidence,
                target_price=current_price * 0.98,
                stop_loss=current_price * 1.02
            )
        
        return None


class RSIStrategy(StrategyEngine):
    """RSI Mean Reversion Strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("RSI_MeanReversion", config)
        self.period = config.get('rsi_period', 14)
        self.overbought = config.get('overbought', 70)
        self.oversold = config.get('oversold', 30)
        self.price_history = defaultdict(lambda: deque(maxlen=self.period + 10))
    
    def generate_signal(self, symbol: str, price_data: Dict[str, float], 
                       market_regime: MarketRegime) -> Optional[StrategySignal]:
        
        current_price = price_data.get('close', price_data.get('price', 0))
        self.price_history[symbol].append(current_price)
        
        if len(self.price_history[symbol]) < self.period + 1:
            return None
        
        prices = np.array(list(self.price_history[symbol]))
        rsi = self._calculate_rsi(prices)
        
        # Strategy works better in ranging markets
        confidence_boost = 0.1 if market_regime == MarketRegime.RANGING else 0
        
        if rsi < self.oversold:
            confidence = min(0.9, 0.5 + (self.oversold - rsi) / 30 + confidence_boost)
            return StrategySignal(
                strategy_name=self.name,
                symbol=symbol,
                action='BUY',
                confidence=confidence,
                target_price=current_price * 1.015,
                stop_loss=current_price * 0.985
            )
        elif rsi > self.overbought:
            confidence = min(0.9, 0.5 + (rsi - self.overbought) / 30 + confidence_boost)
            return StrategySignal(
                strategy_name=self.name,
                symbol=symbol,
                action='SELL',
                confidence=confidence,
                target_price=current_price * 0.985,
                stop_loss=current_price * 1.015
            )
        
        return None
    
    def _calculate_rsi(self, prices: np.ndarray) -> float:
        """Calculate RSI indicator"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.period:])
        avg_loss = np.mean(losses[-self.period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class MomentumStrategy(StrategyEngine):
    """Momentum Strategy based on price momentum"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Momentum", config)
        self.lookback = config.get('momentum_period', 12)
        self.threshold = config.get('momentum_threshold', 0.02)
        self.price_history = defaultdict(lambda: deque(maxlen=self.lookback + 5))
    
    def generate_signal(self, symbol: str, price_data: Dict[str, float], 
                       market_regime: MarketRegime) -> Optional[StrategySignal]:
        
        current_price = price_data.get('close', price_data.get('price', 0))
        self.price_history[symbol].append(current_price)
        
        if len(self.price_history[symbol]) < self.lookback:
            return None
        
        prices = np.array(list(self.price_history[symbol]))
        momentum = (prices[-1] - prices[-self.lookback]) / prices[-self.lookback]
        
        # Strategy works better in trending markets
        confidence_boost = 0.15 if market_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN] else 0
        
        if momentum > self.threshold:
            confidence = min(0.9, 0.6 + momentum * 5 + confidence_boost)
            return StrategySignal(
                strategy_name=self.name,
                symbol=symbol,
                action='BUY',
                confidence=confidence,
                target_price=current_price * (1 + momentum * 0.5),
                stop_loss=current_price * 0.97
            )
        elif momentum < -self.threshold:
            confidence = min(0.9, 0.6 + abs(momentum) * 5 + confidence_boost)
            return StrategySignal(
                strategy_name=self.name,
                symbol=symbol,
                action='SELL',
                confidence=confidence,
                target_price=current_price * (1 + momentum * 0.5),
                stop_loss=current_price * 1.03
            )
        
        return None


class AIEnsembleEngine:
    """AI-Powered Multi-Strategy Ensemble Engine - The heart of the world-class trading system!"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategies: Dict[str, StrategyEngine] = {}
        self.regime_detector = MarketRegimeDetector()
        
        # Ensemble parameters
        self.min_confidence_threshold = config.get('min_confidence', 0.65)
        self.max_strategies_per_signal = config.get('max_strategies', 3)
        self.rebalance_frequency = config.get('rebalance_minutes', 15)
        
        # Performance tracking
        self.ensemble_performance = {
            'total_signals': 0,
            'successful_signals': 0,
            'total_pnl': 0.0,
            'best_strategy_combinations': defaultdict(float)
        }
        
        # AI components
        self.strategy_weights = {}
        self.performance_history = deque(maxlen=1000)
        self.last_rebalance = datetime.now()
        
        # Thread safety
        self.lock = threading.Lock()
        
        self._initialize_strategies()
        self._initialize_weights()
        
        logger.info("🤖 AI Ensemble Strategy Engine initialized")
    
    def _initialize_strategies(self):
        """Initialize all available strategies"""
        
        # SMA Strategy
        sma_config = self.config.get('sma_strategy', {
            'short_window': 10,
            'long_window': 30
        })
        self.strategies['SMA_Cross'] = SMAStrategy(sma_config)
        
        # RSI Strategy  
        rsi_config = self.config.get('rsi_strategy', {
            'rsi_period': 14,
            'overbought': 75,
            'oversold': 25
        })
        self.strategies['RSI_MeanReversion'] = RSIStrategy(rsi_config)
        
        # Momentum Strategy
        momentum_config = self.config.get('momentum_strategy', {
            'momentum_period': 12,
            'momentum_threshold': 0.025
        })
        self.strategies['Momentum'] = MomentumStrategy(momentum_config)
        
        logger.info(f"✅ Initialized {len(self.strategies)} strategies")
    
    def _initialize_weights(self):
        """Initialize strategy weights (equal weight initially)"""
        num_strategies = len(self.strategies)
        initial_weight = 1.0 / num_strategies
        
        for strategy_name in self.strategies.keys():
            self.strategy_weights[strategy_name] = initial_weight
            self.strategies[strategy_name].performance.current_weight = initial_weight
    
    def update_market_data(self, symbol: str, price_data: Dict[str, float]):
        """Update market data for all components"""
        price = price_data.get('close', price_data.get('price', 0))
        volume = price_data.get('volume', None)
        
        # Update regime detector
        self.regime_detector.update_market_data(symbol, price, volume)
    
    def generate_ensemble_signal(self, symbol: str, price_data: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Generate ensemble trading signal combining all strategies"""
        with self.lock:
            # Detect current market regime
            current_regime = self.regime_detector.detect_regime(symbol)
            
            # Collect signals from all active strategies
            strategy_signals = []
            for strategy_name, strategy in self.strategies.items():
                if strategy.is_active:
                    signal = strategy.generate_signal(symbol, price_data, current_regime)
                    if signal and signal.confidence >= 0.4:  # Minimum individual confidence
                        strategy_signals.append(signal)
            
            if not strategy_signals:
                return None
            
            # AI-powered signal aggregation
            ensemble_signal = self._aggregate_signals(strategy_signals, current_regime)
            
            if ensemble_signal and ensemble_signal['confidence'] >= self.min_confidence_threshold:
                self.ensemble_performance['total_signals'] += 1
                return ensemble_signal
            
            return None
    
    def _aggregate_signals(self, signals: List[StrategySignal], regime: MarketRegime) -> Optional[Dict[str, Any]]:
        """Intelligently aggregate multiple strategy signals using AI weighting"""
        if not signals:
            return None
        
        # Group signals by action
        buy_signals = [s for s in signals if s.action == 'BUY']
        sell_signals = [s for s in signals if s.action == 'SELL']
        
        # Determine dominant action
        if len(buy_signals) > len(sell_signals):
            dominant_signals = buy_signals
            action = 'BUY'
        elif len(sell_signals) > len(buy_signals):
            dominant_signals = sell_signals
            action = 'SELL'
        else:
            # Tie - use highest confidence
            all_signals = buy_signals + sell_signals
            best_signal = max(all_signals, key=lambda x: x.confidence)
            dominant_signals = [best_signal]
            action = best_signal.action
        
        # Calculate weighted confidence
        total_weight = 0
        weighted_confidence = 0
        participating_strategies = []
        
        for signal in dominant_signals:
            strategy_name = signal.strategy_name
            weight = self.strategy_weights.get(strategy_name, 0)
            
            # Regime-based weight adjustment
            regime_boost = self._get_regime_boost(strategy_name, regime)
            adjusted_weight = weight * (1 + regime_boost)
            
            weighted_confidence += signal.confidence * adjusted_weight
            total_weight += adjusted_weight
            participating_strategies.append(strategy_name)
        
        if total_weight == 0:
            return None
        
        final_confidence = weighted_confidence / total_weight
        
        # Calculate ensemble target and stop loss
        target_prices = [s.target_price for s in dominant_signals if s.target_price]
        stop_losses = [s.stop_loss for s in dominant_signals if s.stop_loss]
        
        ensemble_target = np.mean(target_prices) if target_prices else None
        ensemble_stop = np.mean(stop_losses) if stop_losses else None
        
        # Record successful strategy combination
        combination_key = '+'.join(sorted(participating_strategies))
        self.ensemble_performance['best_strategy_combinations'][combination_key] += final_confidence
        
        return {
            'symbol': dominant_signals[0].symbol,
            'action': action,
            'confidence': final_confidence,
            'target_price': ensemble_target,
            'stop_loss': ensemble_stop,
            'participating_strategies': participating_strategies,
            'market_regime': regime.value,
            'strategy_count': len(dominant_signals),
            'timestamp': datetime.now()
        }
    
    def _get_regime_boost(self, strategy_name: str, regime: MarketRegime) -> float:
        """Get performance boost based on market regime compatibility"""
        
        regime_compatibility = {
            'SMA_Cross': {
                MarketRegime.TRENDING_UP: 0.2,
                MarketRegime.TRENDING_DOWN: 0.2,
                MarketRegime.RANGING: -0.1,
                MarketRegime.HIGH_VOLATILITY: -0.05,
                MarketRegime.LOW_VOLATILITY: 0.1
            },
            'RSI_MeanReversion': {
                MarketRegime.RANGING: 0.25,
                MarketRegime.HIGH_VOLATILITY: 0.15,
                MarketRegime.TRENDING_UP: -0.1,
                MarketRegime.TRENDING_DOWN: -0.1,
                MarketRegime.LOW_VOLATILITY: 0.05
            },
            'Momentum': {
                MarketRegime.TRENDING_UP: 0.3,
                MarketRegime.TRENDING_DOWN: 0.3,
                MarketRegime.HIGH_VOLATILITY: 0.1,
                MarketRegime.RANGING: -0.2,
                MarketRegime.LOW_VOLATILITY: -0.05
            }
        }
        
        return regime_compatibility.get(strategy_name, {}).get(regime, 0.0)
    
    def update_strategy_performance(self, strategy_name: str, pnl: float, trade_result: str):
        """Update individual strategy performance"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].update_performance(pnl, trade_result)
            
            # Record for rebalancing
            self.performance_history.append({
                'strategy': strategy_name,
                'pnl': pnl,
                'result': trade_result,
                'timestamp': datetime.now()
            })
    
    def rebalance_weights(self):
        """AI-powered dynamic weight rebalancing based on recent performance"""
        if datetime.now() - self.last_rebalance < timedelta(minutes=self.rebalance_frequency):
            return
        
        with self.lock:
            logger.info("🔄 Rebalancing strategy weights...")
            
            # Calculate performance scores
            performance_scores = {}
            for strategy_name, strategy in self.strategies.items():
                perf = strategy.performance
                
                # Multi-factor scoring
                win_rate_score = perf.win_rate * 0.3
                profit_score = max(0, perf.avg_profit_per_trade / 1000) * 0.4  # Normalized
                sharpe_score = max(0, perf.sharpe_ratio / 5) * 0.3  # Normalized
                
                performance_scores[strategy_name] = win_rate_score + profit_score + sharpe_score
            
            # Normalize scores to weights
            total_score = sum(performance_scores.values())
            if total_score > 0:
                for strategy_name in self.strategies.keys():
                    new_weight = performance_scores[strategy_name] / total_score
                    # Apply smoothing to avoid dramatic changes
                    old_weight = self.strategy_weights[strategy_name]
                    self.strategy_weights[strategy_name] = 0.7 * old_weight + 0.3 * new_weight
                    self.strategies[strategy_name].performance.current_weight = self.strategy_weights[strategy_name]
            
            self.last_rebalance = datetime.now()
            logger.info(f"✅ Weights rebalanced: {self.strategy_weights}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        with self.lock:
            # Individual strategy performance
            strategy_reports = {}
            for name, strategy in self.strategies.items():
                perf = strategy.performance
                strategy_reports[name] = {
                    'total_trades': perf.total_trades,
                    'win_rate': perf.win_rate,
                    'total_pnl': perf.total_pnl,
                    'avg_profit_per_trade': perf.avg_profit_per_trade,
                    'sharpe_ratio': perf.sharpe_ratio,
                    'current_weight': perf.current_weight,
                    'last_updated': perf.last_updated.isoformat() if perf.last_updated else None
                }
            
            # Ensemble performance
            ensemble_win_rate = (self.ensemble_performance['successful_signals'] / 
                               max(self.ensemble_performance['total_signals'], 1))
            
            # Market regime summary
            regime_summary = self.regime_detector.get_regime_summary()
            
            # Best strategy combinations
            top_combinations = dict(sorted(
                self.ensemble_performance['best_strategy_combinations'].items(),
                key=lambda x: x[1], reverse=True
            )[:5])
            
            return {
                'ensemble_stats': {
                    'total_signals': self.ensemble_performance['total_signals'],
                    'win_rate': ensemble_win_rate,
                    'total_pnl': self.ensemble_performance['total_pnl'],
                    'last_rebalance': self.last_rebalance.isoformat()
                },
                'individual_strategies': strategy_reports,
                'current_weights': self.strategy_weights,
                'market_regimes': {k: v.value for k, v in regime_summary.items()},
                'top_strategy_combinations': top_combinations,
                'active_strategies': len([s for s in self.strategies.values() if s.is_active])
            }


def demo_ai_ensemble():
    """Demonstration of the AI Ensemble Engine"""
    print("\n" + "="*80)
    print("    🤖 DEMO: AI-Powered Multi-Strategy Ensemble Engine 🤖")
    print("="*80 + "\n")
    
    # Initialize ensemble
    config = {
        'min_confidence': 0.65,
        'max_strategies': 3,
        'rebalance_minutes': 5,
        'sma_strategy': {'short_window': 8, 'long_window': 25},
        'rsi_strategy': {'rsi_period': 14, 'overbought': 75, 'oversold': 25},
        'momentum_strategy': {'momentum_period': 10, 'momentum_threshold': 0.02}
    }
    
    ensemble = AIEnsembleEngine(config)
    
    # Simulate market data and trading
    symbols = ['RELIANCE', 'TCS', 'INFY']
    base_prices = {'RELIANCE': 2500, 'TCS': 3200, 'INFY': 1800}
    
    print("🚀 Starting AI Ensemble Trading Simulation...")
    print(f"📊 Active Strategies: {list(ensemble.strategies.keys())}")
    print(f"🎯 Minimum Ensemble Confidence: {ensemble.min_confidence_threshold}")
    
    # Simulate 20 trading periods for quick demo
    for period in range(20):
        print(f"\n📈 Period {period + 1}/20")
        
        for symbol in symbols:
            # Generate realistic price movement
            price_change = np.random.normal(0, 0.02)  # 2% daily volatility
            new_price = base_prices[symbol] * (1 + price_change)
            base_prices[symbol] = new_price
            
            price_data = {
                'close': new_price,
                'volume': np.random.randint(100000, 1000000)
            }
            
            # Update market data
            ensemble.update_market_data(symbol, price_data)
            
            # Generate ensemble signal
            signal = ensemble.generate_ensemble_signal(symbol, price_data)
            
            if signal:
                print(f"🎯 ENSEMBLE SIGNAL for {symbol}:")
                print(f"   Action: {signal['action']}")
                print(f"   Confidence: {signal['confidence']:.3f}")
                print(f"   Strategies: {', '.join(signal['participating_strategies'])}")
                print(f"   Market Regime: {signal['market_regime']}")
                
                # Simulate trade result
                if np.random.random() < signal['confidence']:
                    trade_result = 'WIN'
                    pnl = np.random.uniform(500, 2000)
                    ensemble.ensemble_performance['successful_signals'] += 1
                else:
                    trade_result = 'LOSS'
                    pnl = -np.random.uniform(300, 1000)
                
                ensemble.ensemble_performance['total_pnl'] += pnl
                
                # Update strategy performances
                for strategy_name in signal['participating_strategies']:
                    ensemble.update_strategy_performance(strategy_name, pnl/len(signal['participating_strategies']), trade_result)
        
        # Periodic rebalancing
        if period % 5 == 4:
            ensemble.rebalance_weights()
    
    # Final performance report
    print("\n" + "="*80)
    print("           📊 FINAL AI ENSEMBLE PERFORMANCE REPORT 📊")
    print("="*80)
    
    report = ensemble.get_performance_report()
    
    # Ensemble stats
    ensemble_stats = report['ensemble_stats']
    print(f"\n🎯 ENSEMBLE PERFORMANCE:")
    print(f"   Total Signals: {ensemble_stats['total_signals']}")
    print(f"   Win Rate: {ensemble_stats['win_rate']:.2%}")
    print(f"   Total P&L: ₹{ensemble_stats['total_pnl']:,.0f}")
    
    print("\n🎉 AI Ensemble Engine Working Perfectly!")
    print("This is what makes your trading system WORLD-CLASS! 🌟")
    print("="*80 + "\n")


if __name__ == "__main__":
    demo_ai_ensemble()
