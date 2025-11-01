"""
🔥 BAMBHORIA QUANTUM ZERODHA API INTEGRATION 🔥
===================================================
Ultimate God-Eye V56 + Zerodha Live Trading Integration
Domain: bambhoriaquantum.in
===================================================

ZERODHA API FEATURES:
✅ Live Market Data Integration
✅ Real-time Order Execution
✅ Portfolio Management
✅ Historical Data Access
✅ WebSocket Streaming
✅ Risk Management Integration
✅ Position Tracking
✅ P&L Monitoring
"""

import os
import json
import time
import logging
import threading
import requests
import websocket
import hashlib
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bambhoria_quantum_zerodha.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ZerodhaConfig:
    """Zerodha API Configuration"""
    # API Credentials (को environment variables से लेना चाहिए)
    api_key: str = "6yif8pldcc4t877p"  # Your Zerodha API key
    api_secret: str = "aghwfgchrqvtqecl7vl9obr86g9dca5j"  # Your Zerodha API secret
    redirect_url: str = "https://bambhoriaquantum.in/callback"
    
    # API URLs
    base_url: str = "https://api.kite.trade"
    login_url: str = "https://kite.trade/connect/login"
    
    # Trading Configuration
    enabled_instruments: List[str] = field(default_factory=lambda: [
        'NSE:NIFTY50', 'NSE:BANKNIFTY', 'NSE:RELIANCE', 'NSE:TCS', 'NSE:INFY',
        'NSE:HDFCBANK', 'NSE:ICICIBANK', 'NSE:SBIN', 'NSE:ITC', 'NSE:HINDUNILVR'
    ])
    
    # Risk Management
    max_position_size: float = 100000.0  # Max position size in INR
    max_daily_loss: float = 10000.0  # Max daily loss in INR
    stop_loss_percentage: float = 2.0  # Stop loss percentage
    take_profit_percentage: float = 5.0  # Take profit percentage
    
    # Domain Configuration
    domain: str = "bambhoriaquantum.in"
    ssl_enabled: bool = True
    webhook_endpoint: str = "/api/webhook/zerodha"

class ZerodhaAPIClient:
    """Bambhoria Quantum Zerodha API Client"""
    
    def __init__(self, config: ZerodhaConfig):
        self.config = config
        self.access_token = None
        self.public_token = None
        self.user_id = None
        self.session = requests.Session()
        self.ws = None
        self.is_connected = False
        self.subscribed_tokens = set()
        
        # Market data storage
        self.live_quotes = {}
        self.historical_data = {}
        self.order_book = []
        self.positions = {}
        self.portfolio = {}
        
        logger.info("🔥 Bambhoria Quantum Zerodha API Client initialized")
        logger.info(f"🌐 Domain: {self.config.domain}")
    
    def get_login_url(self) -> str:
        """Get Zerodha login URL"""
        params = {
            'api_key': self.config.api_key,
            'v': 3
        }
        login_url = f"{self.config.login_url}?{urllib.parse.urlencode(params)}"
        logger.info(f"🔗 Login URL: {login_url}")
        return login_url
    
    def generate_session(self, request_token: str) -> Dict[str, Any]:
        """Generate access token from request token"""
        try:
            url = f"{self.config.base_url}/session/token"
            
            # Generate checksum
            checksum_data = f"{self.config.api_key}{request_token}{self.config.api_secret}"
            checksum = hashlib.sha256(checksum_data.encode()).hexdigest()
            
            data = {
                'api_key': self.config.api_key,
                'request_token': request_token,
                'checksum': checksum
            }
            
            response = self.session.post(url, data=data)
            response.raise_for_status()
            
            session_data = response.json()
            
            if session_data['status'] == 'success':
                self.access_token = session_data['data']['access_token']
                self.public_token = session_data['data']['public_token']
                self.user_id = session_data['data']['user_id']
                
                # Set authorization header
                self.session.headers.update({
                    'Authorization': f'token {self.config.api_key}:{self.access_token}',
                    'X-Kite-Version': '3'
                })
                
                logger.info("✅ Zerodha session generated successfully")
                logger.info(f"👤 User ID: {self.user_id}")
                
                return session_data['data']
            else:
                raise Exception(f"Session generation failed: {session_data}")
                
        except Exception as e:
            logger.error(f"❌ Error generating session: {e}")
            raise
    
    def get_profile(self) -> Dict[str, Any]:
        """Get user profile"""
        try:
            url = f"{self.config.base_url}/user/profile"
            response = self.session.get(url)
            response.raise_for_status()
            
            profile_data = response.json()
            if profile_data['status'] == 'success':
                logger.info("✅ Profile fetched successfully")
                return profile_data['data']
            else:
                raise Exception(f"Profile fetch failed: {profile_data}")
                
        except Exception as e:
            logger.error(f"❌ Error fetching profile: {e}")
            raise
    
    def get_instruments(self, exchange: str = "NSE") -> List[Dict[str, Any]]:
        """Get instruments list"""
        try:
            url = f"{self.config.base_url}/instruments/{exchange}"
            response = self.session.get(url)
            response.raise_for_status()
            
            # Parse CSV data
            import io
            instruments_df = pd.read_csv(io.StringIO(response.text))
            instruments = instruments_df.to_dict('records')
            
            logger.info(f"✅ Fetched {len(instruments)} instruments from {exchange}")
            return instruments
            
        except Exception as e:
            logger.error(f"❌ Error fetching instruments: {e}")
            raise
    
    def get_quote(self, instruments: List[str]) -> Dict[str, Any]:
        """Get live quotes"""
        try:
            url = f"{self.config.base_url}/quote"
            params = {'i': instruments}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            quote_data = response.json()
            if quote_data['status'] == 'success':
                self.live_quotes.update(quote_data['data'])
                logger.info(f"✅ Live quotes updated for {len(instruments)} instruments")
                return quote_data['data']
            else:
                raise Exception(f"Quote fetch failed: {quote_data}")
                
        except Exception as e:
            logger.error(f"❌ Error fetching quotes: {e}")
            raise
    
    def get_historical_data(self, instrument_token: str, from_date: str, to_date: str, 
                           interval: str = "minute") -> List[Dict[str, Any]]:
        """Get historical data"""
        try:
            url = f"{self.config.base_url}/instruments/historical/{instrument_token}/{interval}"
            params = {
                'from': from_date,
                'to': to_date
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            historical_data = response.json()
            if historical_data['status'] == 'success':
                self.historical_data[instrument_token] = historical_data['data']['candles']
                logger.info(f"✅ Historical data fetched for {instrument_token}")
                return historical_data['data']['candles']
            else:
                raise Exception(f"Historical data fetch failed: {historical_data}")
                
        except Exception as e:
            logger.error(f"❌ Error fetching historical data: {e}")
            raise
    
    def place_order(self, tradingsymbol: str, exchange: str, transaction_type: str,
                   quantity: int, order_type: str = "MARKET", product: str = "MIS",
                   price: float = None, trigger_price: float = None,
                   validity: str = "DAY", disclosed_quantity: int = None,
                   squareoff: float = None, stoploss: float = None,
                   trailing_stoploss: float = None, variety: str = "regular") -> Dict[str, Any]:
        """Place order"""
        try:
            url = f"{self.config.base_url}/orders/{variety}"
            
            data = {
                'tradingsymbol': tradingsymbol,
                'exchange': exchange,
                'transaction_type': transaction_type,
                'quantity': quantity,
                'order_type': order_type,
                'product': product,
                'validity': validity
            }
            
            # Add optional parameters
            if price:
                data['price'] = price
            if trigger_price:
                data['trigger_price'] = trigger_price
            if disclosed_quantity:
                data['disclosed_quantity'] = disclosed_quantity
            if squareoff:
                data['squareoff'] = squareoff
            if stoploss:
                data['stoploss'] = stoploss
            if trailing_stoploss:
                data['trailing_stoploss'] = trailing_stoploss
            
            response = self.session.post(url, data=data)
            response.raise_for_status()
            
            order_response = response.json()
            if order_response['status'] == 'success':
                order_id = order_response['data']['order_id']
                self.order_book.append({
                    'order_id': order_id,
                    'tradingsymbol': tradingsymbol,
                    'transaction_type': transaction_type,
                    'quantity': quantity,
                    'order_type': order_type,
                    'timestamp': datetime.now()
                })
                
                logger.info(f"✅ Order placed successfully: {order_id}")
                logger.info(f"   Symbol: {tradingsymbol}")
                logger.info(f"   Type: {transaction_type}")
                logger.info(f"   Quantity: {quantity}")
                
                return order_response['data']
            else:
                raise Exception(f"Order placement failed: {order_response}")
                
        except Exception as e:
            logger.error(f"❌ Error placing order: {e}")
            raise
    
    def modify_order(self, order_id: str, **kwargs) -> Dict[str, Any]:
        """Modify existing order"""
        try:
            url = f"{self.config.base_url}/orders/regular/{order_id}"
            
            response = self.session.put(url, data=kwargs)
            response.raise_for_status()
            
            modify_response = response.json()
            if modify_response['status'] == 'success':
                logger.info(f"✅ Order modified successfully: {order_id}")
                return modify_response['data']
            else:
                raise Exception(f"Order modification failed: {modify_response}")
                
        except Exception as e:
            logger.error(f"❌ Error modifying order: {e}")
            raise
    
    def cancel_order(self, order_id: str, variety: str = "regular") -> Dict[str, Any]:
        """Cancel order"""
        try:
            url = f"{self.config.base_url}/orders/{variety}/{order_id}"
            
            response = self.session.delete(url)
            response.raise_for_status()
            
            cancel_response = response.json()
            if cancel_response['status'] == 'success':
                logger.info(f"✅ Order cancelled successfully: {order_id}")
                return cancel_response['data']
            else:
                raise Exception(f"Order cancellation failed: {cancel_response}")
                
        except Exception as e:
            logger.error(f"❌ Error cancelling order: {e}")
            raise
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """Get all orders"""
        try:
            url = f"{self.config.base_url}/orders"
            response = self.session.get(url)
            response.raise_for_status()
            
            orders_data = response.json()
            if orders_data['status'] == 'success':
                logger.info(f"✅ Fetched {len(orders_data['data'])} orders")
                return orders_data['data']
            else:
                raise Exception(f"Orders fetch failed: {orders_data}")
                
        except Exception as e:
            logger.error(f"❌ Error fetching orders: {e}")
            raise
    
    def get_positions(self) -> Dict[str, Any]:
        """Get positions"""
        try:
            url = f"{self.config.base_url}/portfolio/positions"
            response = self.session.get(url)
            response.raise_for_status()
            
            positions_data = response.json()
            if positions_data['status'] == 'success':
                self.positions = positions_data['data']
                logger.info("✅ Positions updated")
                return positions_data['data']
            else:
                raise Exception(f"Positions fetch failed: {positions_data}")
                
        except Exception as e:
            logger.error(f"❌ Error fetching positions: {e}")
            raise
    
    def get_holdings(self) -> List[Dict[str, Any]]:
        """Get holdings"""
        try:
            url = f"{self.config.base_url}/portfolio/holdings"
            response = self.session.get(url)
            response.raise_for_status()
            
            holdings_data = response.json()
            if holdings_data['status'] == 'success':
                logger.info(f"✅ Fetched {len(holdings_data['data'])} holdings")
                return holdings_data['data']
            else:
                raise Exception(f"Holdings fetch failed: {holdings_data}")
                
        except Exception as e:
            logger.error(f"❌ Error fetching holdings: {e}")
            raise
    
    def start_websocket(self, tokens: List[int]):
        """Start WebSocket for live data"""
        def on_message(ws, message):
            try:
                # Parse binary tick data
                import struct
                
                # Basic tick parsing (simplified)
                if len(message) >= 8:
                    token = struct.unpack('!I', message[:4])[0]
                    ltp = struct.unpack('!I', message[4:8])[0] / 100.0
                    
                    self.live_quotes[token] = {
                        'last_price': ltp,
                        'timestamp': datetime.now()
                    }
                    
                    if len(self.live_quotes) % 100 == 0:
                        logger.info(f"📊 Live data: {len(self.live_quotes)} instruments updated")
                        
            except Exception as e:
                logger.error(f"❌ WebSocket message error: {e}")
        
        def on_error(ws, error):
            logger.error(f"❌ WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info("🔌 WebSocket connection closed")
            self.is_connected = False
        
        def on_open(ws):
            logger.info("✅ WebSocket connection opened")
            self.is_connected = True
            
            # Subscribe to tokens
            import struct
            
            # Subscribe message format (simplified)
            subscribe_message = b''
            for token in tokens:
                subscribe_message += struct.pack('!I', token)
            
            ws.send(subscribe_message)
            logger.info(f"📡 Subscribed to {len(tokens)} instruments")
        
        # WebSocket URL
        ws_url = f"wss://ws.kite.trade?api_key={self.config.api_key}&access_token={self.access_token}"
        
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Start WebSocket in separate thread
        def run_websocket():
            self.ws.run_forever()
        
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()
        
        logger.info("🚀 WebSocket started for live data streaming")

class BambhoriaQuantumZerodhaIntegration:
    """Main Integration Class for Bambhoria Quantum + Zerodha"""
    
    def __init__(self, config: ZerodhaConfig):
        self.config = config
        self.zerodha_client = ZerodhaAPIClient(config)
        self.is_trading_active = False
        self.risk_manager = None
        self.performance_tracker = {}
        
        logger.info("🔥 Bambhoria Quantum Zerodha Integration initialized")
        logger.info(f"🌐 Domain: {config.domain}")
    
    def authenticate(self, request_token: str):
        """Authenticate with Zerodha"""
        try:
            # Generate session
            session_data = self.zerodha_client.generate_session(request_token)
            
            # Get profile
            profile = self.zerodha_client.get_profile()
            
            logger.info("✅ Zerodha authentication successful")
            logger.info(f"👤 User: {profile.get('user_name', 'Unknown')}")
            logger.info(f"📧 Email: {profile.get('email', 'Unknown')}")
            
            return {
                'status': 'success',
                'session_data': session_data,
                'profile': profile
            }
            
        except Exception as e:
            logger.error(f"❌ Authentication failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def start_live_trading(self):
        """Start live trading integration"""
        try:
            logger.info("🚀 Starting Bambhoria Quantum Live Trading...")
            
            # Get instruments
            instruments = self.zerodha_client.get_instruments("NSE")
            logger.info(f"📊 Loaded {len(instruments)} NSE instruments")
            
            # Get enabled instrument tokens
            enabled_tokens = []
            for instrument in instruments:
                if f"NSE:{instrument['tradingsymbol']}" in self.config.enabled_instruments:
                    enabled_tokens.append(instrument['instrument_token'])
            
            # Start WebSocket for live data
            if enabled_tokens:
                self.zerodha_client.start_websocket(enabled_tokens)
            
            # Initialize risk manager
            self._initialize_risk_manager()
            
            # Start monitoring threads
            self._start_monitoring_threads()
            
            self.is_trading_active = True
            logger.info("✅ Bambhoria Quantum Live Trading started successfully")
            
        except Exception as e:
            logger.error(f"❌ Error starting live trading: {e}")
            raise
    
    def _initialize_risk_manager(self):
        """Initialize risk management"""
        self.risk_manager = {
            'max_position_size': self.config.max_position_size,
            'max_daily_loss': self.config.max_daily_loss,
            'current_daily_pnl': 0.0,
            'active_positions': {},
            'stop_losses': {},
            'take_profits': {}
        }
        logger.info("✅ Risk manager initialized")
    
    def _start_monitoring_threads(self):
        """Start monitoring threads"""
        def position_monitor():
            while self.is_trading_active:
                try:
                    # Update positions
                    positions = self.zerodha_client.get_positions()
                    
                    # Update portfolio
                    holdings = self.zerodha_client.get_holdings()
                    
                    # Calculate P&L
                    self._update_performance_metrics()
                    
                    time.sleep(30)  # Update every 30 seconds
                    
                except Exception as e:
                    logger.error(f"❌ Position monitor error: {e}")
                    time.sleep(60)
        
        def risk_monitor():
            while self.is_trading_active:
                try:
                    # Check daily loss limits
                    if self.risk_manager['current_daily_pnl'] < -self.config.max_daily_loss:
                        logger.warning("⚠️ Daily loss limit reached - stopping trading")
                        self._emergency_stop_trading()
                    
                    # Check individual position sizes
                    for symbol, position in self.risk_manager['active_positions'].items():
                        if abs(position['value']) > self.config.max_position_size:
                            logger.warning(f"⚠️ Position size limit exceeded for {symbol}")
                            self._reduce_position(symbol)
                    
                    time.sleep(10)  # Check every 10 seconds
                    
                except Exception as e:
                    logger.error(f"❌ Risk monitor error: {e}")
                    time.sleep(30)
        
        # Start monitoring threads
        position_thread = threading.Thread(target=position_monitor, daemon=True)
        risk_thread = threading.Thread(target=risk_monitor, daemon=True)
        
        position_thread.start()
        risk_thread.start()
        
        logger.info("✅ Monitoring threads started")
    
    def place_intelligent_order(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Place order with intelligent risk management"""
        try:
            symbol = signal['symbol']
            action = signal['action']  # 'BUY' or 'SELL'
            confidence = signal.get('confidence', 0.5)
            
            # Calculate position size based on confidence and risk
            base_quantity = 1
            risk_adjusted_quantity = int(base_quantity * confidence)
            
            # Apply risk limits
            if risk_adjusted_quantity * signal.get('price', 0) > self.config.max_position_size:
                risk_adjusted_quantity = int(self.config.max_position_size / signal.get('price', 1))
            
            # Place order
            order_response = self.zerodha_client.place_order(
                tradingsymbol=symbol,
                exchange="NSE",
                transaction_type=action,
                quantity=risk_adjusted_quantity,
                order_type="MARKET",
                product="MIS"
            )
            
            # Set stop loss and take profit
            if order_response:
                self._set_protective_orders(symbol, action, signal.get('price'), risk_adjusted_quantity)
            
            logger.info(f"✅ Intelligent order placed: {symbol} {action} {risk_adjusted_quantity}")
            
            return {
                'status': 'success',
                'order_id': order_response.get('order_id'),
                'symbol': symbol,
                'action': action,
                'quantity': risk_adjusted_quantity
            }
            
        except Exception as e:
            logger.error(f"❌ Error placing intelligent order: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _set_protective_orders(self, symbol: str, action: str, price: float, quantity: int):
        """Set stop loss and take profit orders"""
        try:
            if action == "BUY":
                # Stop loss below current price
                stop_loss_price = price * (1 - self.config.stop_loss_percentage / 100)
                take_profit_price = price * (1 + self.config.take_profit_percentage / 100)
                
                # Place stop loss order
                self.zerodha_client.place_order(
                    tradingsymbol=symbol,
                    exchange="NSE",
                    transaction_type="SELL",
                    quantity=quantity,
                    order_type="SL",
                    product="MIS",
                    trigger_price=stop_loss_price,
                    price=stop_loss_price * 0.99
                )
                
                # Place take profit order
                self.zerodha_client.place_order(
                    tradingsymbol=symbol,
                    exchange="NSE",
                    transaction_type="SELL",
                    quantity=quantity,
                    order_type="LIMIT",
                    product="MIS",
                    price=take_profit_price
                )
                
            elif action == "SELL":
                # Stop loss above current price
                stop_loss_price = price * (1 + self.config.stop_loss_percentage / 100)
                take_profit_price = price * (1 - self.config.take_profit_percentage / 100)
                
                # Place stop loss order
                self.zerodha_client.place_order(
                    tradingsymbol=symbol,
                    exchange="NSE",
                    transaction_type="BUY",
                    quantity=quantity,
                    order_type="SL",
                    product="MIS",
                    trigger_price=stop_loss_price,
                    price=stop_loss_price * 1.01
                )
                
                # Place take profit order
                self.zerodha_client.place_order(
                    tradingsymbol=symbol,
                    exchange="NSE",
                    transaction_type="BUY",
                    quantity=quantity,
                    order_type="LIMIT",
                    product="MIS",
                    price=take_profit_price
                )
            
            logger.info(f"✅ Protective orders set for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error setting protective orders: {e}")
    
    def _update_performance_metrics(self):
        """Update performance tracking"""
        try:
            # Get current positions
            positions = self.zerodha_client.get_positions()
            
            total_pnl = 0.0
            for position in positions.get('day', []):
                pnl = float(position.get('pnl', 0))
                total_pnl += pnl
            
            self.risk_manager['current_daily_pnl'] = total_pnl
            
            # Update performance tracker
            self.performance_tracker.update({
                'daily_pnl': total_pnl,
                'total_trades': len(self.zerodha_client.order_book),
                'active_positions': len(positions.get('day', [])),
                'last_updated': datetime.now()
            })
            
        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {e}")
    
    def _emergency_stop_trading(self):
        """Emergency stop all trading"""
        try:
            logger.warning("🚨 EMERGENCY STOP - Closing all positions")
            
            # Get all positions
            positions = self.zerodha_client.get_positions()
            
            # Close all day positions
            for position in positions.get('day', []):
                if int(position['quantity']) != 0:
                    # Determine opposite transaction type
                    transaction_type = "SELL" if int(position['quantity']) > 0 else "BUY"
                    
                    self.zerodha_client.place_order(
                        tradingsymbol=position['tradingsymbol'],
                        exchange=position['exchange'],
                        transaction_type=transaction_type,
                        quantity=abs(int(position['quantity'])),
                        order_type="MARKET",
                        product=position['product']
                    )
            
            self.is_trading_active = False
            logger.info("✅ Emergency stop completed")
            
        except Exception as e:
            logger.error(f"❌ Error in emergency stop: {e}")
    
    def _reduce_position(self, symbol: str):
        """Reduce position size for risk management"""
        try:
            # Implementation for reducing specific position
            logger.info(f"⚠️ Reducing position for {symbol}")
            # Add position reduction logic here
            
        except Exception as e:
            logger.error(f"❌ Error reducing position: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            return {
                'trading_active': self.is_trading_active,
                'websocket_connected': self.zerodha_client.is_connected,
                'domain': self.config.domain,
                'user_id': self.zerodha_client.user_id,
                'performance': self.performance_tracker,
                'risk_metrics': self.risk_manager,
                'live_quotes_count': len(self.zerodha_client.live_quotes),
                'order_book_count': len(self.zerodha_client.order_book),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting system status: {e}")
            return {'error': str(e)}
    
    def stop_trading(self):
        """Stop trading gracefully"""
        try:
            logger.info("🛑 Stopping Bambhoria Quantum trading...")
            
            self.is_trading_active = False
            
            if self.zerodha_client.ws:
                self.zerodha_client.ws.close()
            
            logger.info("✅ Trading stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping trading: {e}")

def create_zerodha_config() -> ZerodhaConfig:
    """Create Zerodha configuration"""
    return ZerodhaConfig(
        # Note: API credentials should be set via environment variables
        api_key=os.getenv('ZERODHA_API_KEY', ''),
        api_secret=os.getenv('ZERODHA_API_SECRET', ''),
        redirect_url="https://bambhoriaquantum.in/callback",
        domain="bambhoriaquantum.in",
        enabled_instruments=[
            'NSE:NIFTY50', 'NSE:BANKNIFTY', 'NSE:RELIANCE', 'NSE:TCS', 'NSE:INFY',
            'NSE:HDFCBANK', 'NSE:ICICIBANK', 'NSE:SBIN', 'NSE:ITC', 'NSE:HINDUNILVR'
        ],
        max_position_size=100000.0,
        max_daily_loss=10000.0,
        stop_loss_percentage=2.0,
        take_profit_percentage=5.0
    )

def main():
    """Main function for testing Zerodha integration"""
    print("\n" + "="*80)
    print("🔥 BAMBHORIA QUANTUM ZERODHA API INTEGRATION 🔥")
    print("Domain: bambhoriaquantum.in")
    print("="*80 + "\n")
    
    # Create configuration
    config = create_zerodha_config()
    
    # Initialize integration
    integration = BambhoriaQuantumZerodhaIntegration(config)
    
    print("🔗 Zerodha Login URL:")
    login_url = integration.zerodha_client.get_login_url()
    print(f"   {login_url}")
    
    print("\n📋 Integration Features:")
    print("   ✅ Live Market Data Streaming")
    print("   ✅ Real-time Order Execution")
    print("   ✅ Intelligent Risk Management")
    print("   ✅ Portfolio Monitoring")
    print("   ✅ Stop Loss & Take Profit")
    print("   ✅ Emergency Stop Protocols")
    print("   ✅ Performance Tracking")
    print("   ✅ WebSocket Live Data")
    
    print("\n🌐 Domain Configuration:")
    print(f"   🏠 Domain: {config.domain}")
    print(f"   🔗 Callback URL: {config.redirect_url}")
    print(f"   📡 Webhook: {config.webhook_endpoint}")
    
    print("\n⚠️ Setup Instructions:")
    print("   1. Set environment variables:")
    print("      ZERODHA_API_KEY=your_api_key")
    print("      ZERODHA_API_SECRET=your_api_secret")
    print("   2. Complete authentication via login URL")
    print("   3. Use request_token to generate session")
    print("   4. Start live trading integration")
    
    print("\n✅ Bambhoria Quantum Zerodha Integration Ready!")

if __name__ == "__main__":
    main()