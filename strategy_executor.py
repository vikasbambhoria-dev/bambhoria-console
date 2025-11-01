"""
strategy_executor.py
Bambhoria Strategy Executor - Execute JSON strategies through the signal engine
Author: Vikas Bambhoria
Purpose:
 - Load strategy JSON files from strategies/ folder
 - Parse entry/exit logic and risk parameters
 - Execute trades using adaptive signal engine with strategy rules
"""

import json, os, re
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

class StrategyExecutor:
    def __init__(self, strategy_path: str):
        """Load and parse strategy JSON"""
        self.strategy_path = Path(strategy_path)
        if not self.strategy_path.exists():
            raise FileNotFoundError(f"Strategy not found: {strategy_path}")
        
        with open(self.strategy_path, 'r', encoding='utf-8') as f:
            self.strategy = json.load(f)
        
        self.name = self.strategy.get('name', 'Unnamed')
        self.components = self.strategy.get('components', {})
        self.indicators = self.components.get('indicators', [])
        self.ai_model = self.components.get('ai_model', 'lightgbm_v51')
        self.entry_logic = self.components.get('entry_logic', '')
        self.exit_logic = self.components.get('exit_logic', '')
        self.risk = self.components.get('risk', {})
        self.meta = self.strategy.get('meta', {})
        
        # Parse logic conditions
        self.entry_conditions = self._parse_logic(self.entry_logic)
        self.exit_conditions = self._parse_logic(self.exit_logic)
        
        print(f"✅ Strategy loaded: {self.name}")
        print(f"   Indicators: {', '.join(self.indicators)}")
        print(f"   AI Model: {self.ai_model}")
        print(f"   Entry: {self.entry_logic}")
        print(f"   Exit: {self.exit_logic}")
        print(f"   Risk: Max Loss=${self.risk.get('max_loss', 'N/A')}, "
              f"Max Pos={self.risk.get('max_positions', 'N/A')}")
    
    def _parse_logic(self, logic_str: str):
        """Parse entry/exit logic into executable conditions"""
        conditions = {
            'action': 'HOLD',
            'rules': []
        }
        
        # Extract action (BUY/SELL)
        if 'BUY' in logic_str.upper():
            conditions['action'] = 'BUY'
        elif 'SELL' in logic_str.upper():
            conditions['action'] = 'SELL'
        
        # Extract indicator conditions using regex
        # Pattern: indicator_name operator value
        patterns = [
            r'momentum\s*([><=]+)\s*([\d.]+)',
            r'rsi\s*([><=]+)\s*([\d.]+)',
            r'macd\s*([><=]+)\s*([\d.]+)',
            r'volume_surge\s*([><=]+)\s*([\d.]+)',
            r'pnl\s*([><=]+)\s*-?([\d.]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, logic_str, re.IGNORECASE)
            for match in matches:
                operator, value = match
                indicator = pattern.split(r'\s')[0].replace('\\', '')
                conditions['rules'].append({
                    'indicator': indicator,
                    'operator': operator,
                    'value': float(value)
                })
        
        return conditions
    
    def calculate_indicators(self, tick: dict, historical_data: list = None):
        """Calculate indicator values for the current tick"""
        indicators_values = {}
        price = tick.get('last_price', 0)
        volume = tick.get('volume', 0)
        
        # Calculate momentum if needed
        if 'momentum' in self.indicators or 'momentum_fast5' in self.indicators:
            if historical_data and len(historical_data) >= 5:
                recent_prices = [h.get('last_price', price) for h in historical_data[-5:]]
                momentum = (price - recent_prices[0]) / recent_prices[0]
                indicators_values['momentum'] = momentum
            else:
                indicators_values['momentum'] = 0.0
        
        # Calculate RSI if needed
        if 'rsi' in self.indicators:
            if historical_data and len(historical_data) >= 14:
                prices = [h.get('last_price', price) for h in historical_data[-14:]]
                gains = [max(0, prices[i] - prices[i-1]) for i in range(1, len(prices))]
                losses = [max(0, prices[i-1] - prices[i]) for i in range(1, len(prices))]
                avg_gain = sum(gains) / len(gains) if gains else 0
                avg_loss = sum(losses) / len(losses) if losses else 1
                rs = avg_gain / avg_loss if avg_loss > 0 else 100
                rsi = 100 - (100 / (1 + rs))
                indicators_values['rsi'] = rsi
            else:
                indicators_values['rsi'] = 50.0  # Neutral
        
        # Calculate volume surge if needed
        if 'volume_surge' in self.indicators:
            if historical_data and len(historical_data) >= 10:
                avg_volume = sum([h.get('volume', 0) for h in historical_data[-10:]]) / 10
                surge = volume / avg_volume if avg_volume > 0 else 1.0
                indicators_values['volume_surge'] = surge
            else:
                indicators_values['volume_surge'] = 1.0
        
        return indicators_values
    
    def check_conditions(self, conditions: dict, indicators: dict, current_pnl: float = 0):
        """Check if strategy conditions are met"""
        if not conditions['rules']:
            return False
        
        for rule in conditions['rules']:
            indicator_name = rule['indicator']
            operator = rule['operator']
            threshold = rule['value']
            
            # Get indicator value
            if indicator_name == 'pnl':
                value = current_pnl
            else:
                value = indicators.get(indicator_name, 0)
            
            # Evaluate condition
            if operator == '>':
                if not (value > threshold):
                    return False
            elif operator == '<':
                if not (value < threshold):
                    return False
            elif operator == '>=':
                if not (value >= threshold):
                    return False
            elif operator == '<=':
                if not (value <= threshold):
                    return False
            elif operator == '==':
                if not (abs(value - threshold) < 0.001):
                    return False
        
        return True
    
    def should_enter(self, tick: dict, historical_data: list = None):
        """Check if entry conditions are met"""
        indicators = self.calculate_indicators(tick, historical_data)
        is_met = self.check_conditions(self.entry_conditions, indicators)
        
        if is_met:
            print(f"✅ ENTRY triggered: {self.entry_logic}")
            print(f"   Indicators: {indicators}")
        
        return is_met, self.entry_conditions['action']
    
    def should_exit(self, tick: dict, current_pnl: float, historical_data: list = None):
        """Check if exit conditions are met"""
        indicators = self.calculate_indicators(tick, historical_data)
        is_met = self.check_conditions(self.exit_conditions, indicators, current_pnl)
        
        if is_met:
            print(f"🚪 EXIT triggered: {self.exit_logic}")
            print(f"   Indicators: {indicators}, PnL: ₹{current_pnl:+.2f}")
        
        return is_met, self.exit_conditions['action']
    
    def get_risk_params(self):
        """Get risk management parameters"""
        return {
            'max_loss': self.risk.get('max_loss', 5000),
            'max_positions': self.risk.get('max_positions', 100),
            'cooldown_secs': self.risk.get('cooldown_secs', 10),
            'trailing_stop_pct': self.risk.get('trailing_stop_pct', 0.015),
            'take_profit_rr': self.risk.get('take_profit_rr', 2.0)
        }
    
    def get_expected_performance(self):
        """Get expected strategy performance from meta"""
        return {
            'win_rate': self.meta.get('expected_winrate', 0.5),
            'risk_reward': self.meta.get('expected_risk_reward', 1.5),
            'strategy_id': self.meta.get('strategy_id', 'unknown')
        }


def demo_strategy_executor():
    """Demo: Load and test Alpha Neural Drive strategy"""
    print("🎯 Bambhoria Strategy Executor Demo")
    print("=" * 60)
    
    # Load Alpha Neural Drive
    executor = StrategyExecutor("strategies/Alpha_Neural_Drive.json")
    
    print("\n📊 Expected Performance:")
    perf = executor.get_expected_performance()
    print(f"   Win Rate: {perf['win_rate']*100:.1f}%")
    print(f"   Risk/Reward: {perf['risk_reward']:.1f}")
    
    print("\n🔬 Testing with sample ticks...")
    
    # Simulate historical data
    historical = []
    for i in range(20):
        historical.append({
            'last_price': 2500 + i * 2,
            'volume': 10000 + i * 500
        })
    
    # Test tick with momentum and RSI conditions
    test_tick = {
        'symbol': 'RELIANCE',
        'last_price': 2550,  # 4% momentum from 2500
        'volume': 15000
    }
    
    print(f"\n📈 Test Tick: {test_tick['symbol']} @ ₹{test_tick['last_price']}")
    
    should_enter, action = executor.should_enter(test_tick, historical)
    print(f"   Should Enter: {should_enter} ({action})")
    
    # Test exit with loss
    should_exit, exit_action = executor.should_exit(test_tick, current_pnl=-150, historical_data=historical)
    print(f"   Should Exit: {should_exit} (PnL: -₹150)")
    
    print("\n✅ Strategy executor demo complete")


if __name__ == "__main__":
    demo_strategy_executor()
