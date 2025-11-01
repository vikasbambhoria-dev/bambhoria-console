"""
risk_manager.py
Bambhoria Risk Management System
Comprehensive risk controls for the trading system
"""

import json
import logging
from datetime import datetime, date
from typing import Dict, List, Any, Optional

class RiskManager:
    def __init__(self, 
                 max_daily_loss: float = 5000,
                 max_position_size: int = 100,
                 max_positions_per_symbol: int = 50,
                 volatility_cutoff: float = 25.0,
                 log_path: str = "logs/risk_log.json"):
        
        self.max_daily_loss = max_daily_loss
        self.max_position_size = max_position_size
        self.max_positions_per_symbol = max_positions_per_symbol
        self.volatility_cutoff = volatility_cutoff
        self.log_path = log_path
        
        # Risk tracking
        self.pnl_today = 0.0
        self.positions = {}  # symbol -> quantity
        self.trades = []
        self.last_reset_date = date.today()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("RiskManager")
        
        self.logger.info(f"RiskManager initialized: {self.__dict__}")
    
    def check_risk(self, symbol: str, quantity: int, price: float, action: str) -> Dict[str, Any]:
        """
        Check if a trade is allowed based on risk parameters
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares
            price: Current price
            action: 'BUY' or 'SELL'
            
        Returns:
            Dict with 'allowed' (bool) and 'reason' (str)
        """
        try:
            # Reset daily tracking if new day
            if date.today() != self.last_reset_date:
                self.pnl_today = 0.0
                self.last_reset_date = date.today()
                self.logger.info("Daily risk limits reset")
            
            # Check daily loss limit
            if self.pnl_today <= -self.max_daily_loss:
                return {
                    'allowed': False,
                    'reason': f'Daily loss limit exceeded: ₹{abs(self.pnl_today):.2f}'
                }
            
            # Check position size limit
            current_position = self.positions.get(symbol, 0)
            
            if action == 'BUY':
                new_position = current_position + quantity
            else:  # SELL
                new_position = current_position - quantity
            
            if abs(new_position) > self.max_positions_per_symbol:
                return {
                    'allowed': False,
                    'reason': f'Position limit exceeded for {symbol}: {abs(new_position)}'
                }
            
            # Check individual trade size
            if quantity > self.max_position_size:
                return {
                    'allowed': False,
                    'reason': f'Trade size too large: {quantity} > {self.max_position_size}'
                }
            
            # Check trade value (additional safety)
            trade_value = quantity * price
            if trade_value > 50000:  # 50k per trade limit
                return {
                    'allowed': False,
                    'reason': f'Trade value too high: ₹{trade_value:.2f}'
                }
            
            # All checks passed
            return {
                'allowed': True,
                'reason': 'Risk checks passed'
            }
            
        except Exception as e:
            self.logger.error(f"Risk check error: {e}")
            return {
                'allowed': False,
                'reason': f'Risk check failed: {str(e)}'
            }
    
    def record_trade(self, symbol: str, quantity: int, price: float, action: str, pnl: float = 0):
        """Record a completed trade"""
        try:
            trade = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'action': action,
                'pnl': pnl,
                'value': quantity * price
            }
            
            self.trades.append(trade)
            
            # Update position tracking
            if action == 'BUY':
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            else:  # SELL
                self.positions[symbol] = self.positions.get(symbol, 0) - quantity
            
            # Update daily P&L
            self.pnl_today += pnl
            
            # Log the trade
            self.logger.info(f"Trade recorded: {symbol} {action} {quantity}@₹{price:.2f} P&L=₹{pnl:.2f}")
            
            # Save to disk
            self._save_risk_log()
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status summary"""
        return {
            'daily_pnl': self.pnl_today,
            'daily_loss_limit': self.max_daily_loss,
            'daily_limit_remaining': self.max_daily_loss + self.pnl_today,
            'positions': self.positions.copy(),
            'total_trades_today': len([t for t in self.trades if t['timestamp'].startswith(str(date.today()))]),
            'risk_level': self._calculate_risk_level()
        }
    
    def _calculate_risk_level(self) -> str:
        """Calculate current risk level"""
        loss_ratio = abs(self.pnl_today) / self.max_daily_loss
        
        if loss_ratio > 0.8:
            return "HIGH"
        elif loss_ratio > 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _save_risk_log(self):
        """Save risk log to file"""
        try:
            risk_data = {
                'timestamp': datetime.now().isoformat(),
                'pnl_today': self.pnl_today,
                'positions': self.positions,
                'trades': self.trades[-10:],  # Last 10 trades
                'risk_status': self.get_risk_status()
            }
            
            with open(self.log_path, 'w') as f:
                json.dump(risk_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving risk log: {e}")
