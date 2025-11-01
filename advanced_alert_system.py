"""
🚨 BAMBHORIA GOD-EYE V52 - ADVANCED ALERT & NOTIFICATION SYSTEM 🚨
==================================================================
Intelligent Multi-Channel Alert System
Real-Time Risk Monitoring & Critical Event Notifications
==================================================================
"""

import json
import time
import logging
import threading
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque
import sqlite3
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertCategory(Enum):
    """Alert categories"""
    PERFORMANCE = "performance"
    RISK_MANAGEMENT = "risk_management"
    MARKET_CONDITIONS = "market_conditions"
    SYSTEM_HEALTH = "system_health"
    SENTIMENT = "sentiment"
    TRADING_SIGNAL = "trading_signal"
    ANOMALY = "anomaly"

class NotificationChannel(Enum):
    """Notification delivery channels"""
    CONSOLE = "console"
    WEB_DASHBOARD = "web_dashboard"
    LOG_FILE = "log_file"
    EMAIL = "email"
    SMS = "sms"
    DESKTOP = "desktop"

@dataclass
class Alert:
    """Data class for alerts"""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    category: AlertCategory
    title: str
    message: str
    details: Dict[str, Any]
    action_required: bool
    auto_resolve: bool
    expiry_time: Optional[datetime] = None
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class NotificationRule:
    """Data class for notification rules"""
    name: str
    conditions: Dict[str, Any]
    channels: List[NotificationChannel]
    severity_threshold: AlertSeverity
    cooldown_minutes: int
    enabled: bool = True

class AlertManager:
    """Core alert management system"""
    
    def __init__(self):
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.notification_rules = []
        self.alert_callbacks = []
        
        # Alert counters
        self.alert_counters = {
            'total': 0,
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'info': 0
        }
        
        # Cooldown tracking
        self.last_alerts = {}
        
        logger.info("🚨 Alert Manager initialized")
    
    def create_alert(self, severity: AlertSeverity, category: AlertCategory, 
                    title: str, message: str, details: Dict[str, Any] = None,
                    action_required: bool = False, auto_resolve: bool = True,
                    expiry_minutes: int = None) -> Alert:
        """Create a new alert"""
        
        alert_id = f"{category.value}_{int(time.time())}"
        expiry_time = None
        
        if expiry_minutes:
            expiry_time = datetime.now() + timedelta(minutes=expiry_minutes)
        
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            title=title,
            message=message,
            details=details or {},
            action_required=action_required,
            auto_resolve=auto_resolve,
            expiry_time=expiry_time
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Update counters
        self.alert_counters['total'] += 1
        self.alert_counters[severity.value] += 1
        
        logger.info(f"🚨 Alert created: {severity.value.upper()} - {title}")
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            callback(alert)
        
        return alert
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"✅ Alert acknowledged: {alert_id}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            del self.active_alerts[alert_id]
            logger.info(f"✅ Alert resolved: {alert_id}")
            return True
        return False
    
    def get_active_alerts(self, severity_filter: AlertSeverity = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity"""
        alerts = list(self.active_alerts.values())
        
        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def cleanup_expired_alerts(self) -> int:
        """Clean up expired alerts"""
        now = datetime.now()
        expired_alerts = []
        
        for alert_id, alert in self.active_alerts.items():
            if alert.expiry_time and now > alert.expiry_time:
                expired_alerts.append(alert_id)
        
        for alert_id in expired_alerts:
            self.resolve_alert(alert_id)
        
        return len(expired_alerts)
    
    def register_alert_callback(self, callback: Callable) -> None:
        """Register alert callback"""
        self.alert_callbacks.append(callback)

class RiskMonitor:
    """Risk-based alert monitoring"""
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.risk_thresholds = {
            'max_drawdown': 0.05,  # 5%
            'daily_loss_limit': 0.03,  # 3%
            'position_concentration': 0.25,  # 25% per symbol
            'leverage_limit': 2.0,
            'win_rate_threshold': 0.40  # 40%
        }
        
        # Risk tracking
        self.current_metrics = {}
        self.risk_alerts_sent = set()
        
    def check_risk_metrics(self, metrics: Dict[str, Any]) -> None:
        """Check risk metrics and generate alerts"""
        self.current_metrics = metrics
        
        # Drawdown check
        current_drawdown = abs(metrics.get('max_drawdown', 0))
        if current_drawdown > self.risk_thresholds['max_drawdown']:
            if 'max_drawdown' not in self.risk_alerts_sent:
                self.alert_manager.create_alert(
                    severity=AlertSeverity.CRITICAL,
                    category=AlertCategory.RISK_MANAGEMENT,
                    title="Maximum Drawdown Exceeded",
                    message=f"Current drawdown: {current_drawdown:.1%} exceeds limit of {self.risk_thresholds['max_drawdown']:.1%}",
                    details={'current_drawdown': current_drawdown, 'threshold': self.risk_thresholds['max_drawdown']},
                    action_required=True
                )
                self.risk_alerts_sent.add('max_drawdown')
        else:
            self.risk_alerts_sent.discard('max_drawdown')
        
        # Daily loss limit
        daily_pnl = metrics.get('daily_pnl', 0)
        account_balance = metrics.get('account_balance', 100000)
        daily_loss_pct = abs(daily_pnl) / account_balance if account_balance > 0 else 0
        
        if daily_pnl < 0 and daily_loss_pct > self.risk_thresholds['daily_loss_limit']:
            if 'daily_loss' not in self.risk_alerts_sent:
                self.alert_manager.create_alert(
                    severity=AlertSeverity.HIGH,
                    category=AlertCategory.RISK_MANAGEMENT,
                    title="Daily Loss Limit Approached",
                    message=f"Daily loss: {daily_loss_pct:.1%} approaching limit of {self.risk_thresholds['daily_loss_limit']:.1%}",
                    details={'daily_pnl': daily_pnl, 'loss_percentage': daily_loss_pct},
                    action_required=True
                )
                self.risk_alerts_sent.add('daily_loss')
        else:
            self.risk_alerts_sent.discard('daily_loss')
        
        # Win rate check
        win_rate = metrics.get('win_rate', 1.0)
        if win_rate < self.risk_thresholds['win_rate_threshold']:
            if 'low_win_rate' not in self.risk_alerts_sent:
                self.alert_manager.create_alert(
                    severity=AlertSeverity.MEDIUM,
                    category=AlertCategory.PERFORMANCE,
                    title="Low Win Rate Detected",
                    message=f"Win rate {win_rate:.1%} below threshold of {self.risk_thresholds['win_rate_threshold']:.1%}",
                    details={'win_rate': win_rate, 'threshold': self.risk_thresholds['win_rate_threshold']},
                    action_required=False
                )
                self.risk_alerts_sent.add('low_win_rate')
        else:
            self.risk_alerts_sent.discard('low_win_rate')

class MarketMonitor:
    """Market condition monitoring"""
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.market_thresholds = {
            'volatility_spike': 0.5,  # 50% above normal
            'volume_spike': 3.0,  # 3x normal volume
            'price_gap': 0.05,  # 5% price gap
            'correlation_break': 0.3  # Correlation below 30%
        }
        
        self.market_data_history = {}
        
    def check_market_conditions(self, market_data: Dict[str, Any]) -> None:
        """Check market conditions for alerts"""
        
        for symbol, data in market_data.items():
            current_price = data.get('price', 0)
            current_volume = data.get('volume', 0)
            
            # Store historical data
            if symbol not in self.market_data_history:
                self.market_data_history[symbol] = {
                    'prices': deque(maxlen=100),
                    'volumes': deque(maxlen=100)
                }
            
            history = self.market_data_history[symbol]
            history['prices'].append(current_price)
            history['volumes'].append(current_volume)
            
            # Volume spike detection
            if len(history['volumes']) >= 20:
                avg_volume = sum(list(history['volumes'])[-20:]) / 20
                if avg_volume > 0 and current_volume > avg_volume * self.market_thresholds['volume_spike']:
                    self.alert_manager.create_alert(
                        severity=AlertSeverity.MEDIUM,
                        category=AlertCategory.MARKET_CONDITIONS,
                        title=f"{symbol} Volume Spike Detected",
                        message=f"Volume {current_volume:,} is {current_volume/avg_volume:.1f}x normal",
                        details={'symbol': symbol, 'current_volume': current_volume, 'avg_volume': avg_volume},
                        expiry_minutes=30
                    )
            
            # Price gap detection
            if len(history['prices']) >= 2:
                prev_price = history['prices'][-2]
                if prev_price > 0:
                    price_change = abs(current_price - prev_price) / prev_price
                    if price_change > self.market_thresholds['price_gap']:
                        self.alert_manager.create_alert(
                            severity=AlertSeverity.HIGH,
                            category=AlertCategory.MARKET_CONDITIONS,
                            title=f"{symbol} Significant Price Movement",
                            message=f"Price moved {price_change:.1%} from ₹{prev_price:.2f} to ₹{current_price:.2f}",
                            details={'symbol': symbol, 'price_change': price_change, 'prev_price': prev_price, 'current_price': current_price},
                            expiry_minutes=60
                        )

class PerformanceMonitor:
    """Performance monitoring and alerts"""
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.performance_thresholds = {
            'profit_milestone': [5000, 10000, 25000, 50000, 100000],
            'loss_milestone': [2000, 5000, 10000, 20000],
            'trades_milestone': [50, 100, 200, 500, 1000],
            'win_streak': 10,
            'loss_streak': 5
        }
        
        self.milestones_reached = set()
        
    def check_performance_milestones(self, metrics: Dict[str, Any]) -> None:
        """Check for performance milestones"""
        
        total_pnl = metrics.get('total_pnl', 0)
        trades_count = metrics.get('trades_count', 0)
        
        # Profit milestones
        if total_pnl > 0:
            for milestone in self.performance_thresholds['profit_milestone']:
                if total_pnl >= milestone and f"profit_{milestone}" not in self.milestones_reached:
                    self.alert_manager.create_alert(
                        severity=AlertSeverity.INFO,
                        category=AlertCategory.PERFORMANCE,
                        title=f"Profit Milestone Reached!",
                        message=f"Congratulations! Total profit reached ₹{milestone:,}",
                        details={'milestone': milestone, 'current_pnl': total_pnl},
                        expiry_minutes=120
                    )
                    self.milestones_reached.add(f"profit_{milestone}")
        
        # Loss milestones (warnings)
        if total_pnl < 0:
            for milestone in self.performance_thresholds['loss_milestone']:
                if abs(total_pnl) >= milestone and f"loss_{milestone}" not in self.milestones_reached:
                    self.alert_manager.create_alert(
                        severity=AlertSeverity.MEDIUM,
                        category=AlertCategory.PERFORMANCE,
                        title=f"Loss Milestone Alert",
                        message=f"Total loss reached ₹{milestone:,} - Review strategy",
                        details={'milestone': milestone, 'current_pnl': total_pnl},
                        action_required=True
                    )
                    self.milestones_reached.add(f"loss_{milestone}")
        
        # Trade count milestones
        for milestone in self.performance_thresholds['trades_milestone']:
            if trades_count >= milestone and f"trades_{milestone}" not in self.milestones_reached:
                self.alert_manager.create_alert(
                    severity=AlertSeverity.INFO,
                    category=AlertCategory.PERFORMANCE,
                    title=f"Trading Activity Milestone",
                    message=f"Completed {milestone} trades! System gaining experience.",
                    details={'milestone': milestone, 'trades_count': trades_count}
                )
                self.milestones_reached.add(f"trades_{milestone}")

class NotificationDelivery:
    """Multi-channel notification delivery system"""
    
    def __init__(self):
        self.enabled_channels = [NotificationChannel.CONSOLE, NotificationChannel.LOG_FILE]
        self.notification_history = deque(maxlen=500)
        
        # Channel configurations
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': '',  # To be configured
            'password': '',  # To be configured
            'from_email': '',
            'to_emails': []
        }
        
    def deliver_notification(self, alert: Alert) -> Dict[str, bool]:
        """Deliver notification through enabled channels"""
        delivery_status = {}
        
        # Console notification
        if NotificationChannel.CONSOLE in self.enabled_channels:
            delivery_status['console'] = self._deliver_console(alert)
        
        # Log file notification
        if NotificationChannel.LOG_FILE in self.enabled_channels:
            delivery_status['log_file'] = self._deliver_log_file(alert)
        
        # Web dashboard notification (if enabled)
        if NotificationChannel.WEB_DASHBOARD in self.enabled_channels:
            delivery_status['web_dashboard'] = self._deliver_web_dashboard(alert)
        
        # Email notification (if configured)
        if (NotificationChannel.EMAIL in self.enabled_channels and 
            self.email_config['username'] and alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]):
            delivery_status['email'] = self._deliver_email(alert)
        
        # Store notification
        self.notification_history.append({
            'alert_id': alert.id,
            'timestamp': datetime.now(),
            'delivery_status': delivery_status
        })
        
        return delivery_status
    
    def _deliver_console(self, alert: Alert) -> bool:
        """Deliver console notification"""
        try:
            severity_icons = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.LOW: "🟡",
                AlertSeverity.MEDIUM: "🟠",
                AlertSeverity.HIGH: "🔴",
                AlertSeverity.CRITICAL: "🚨"
            }
            
            icon = severity_icons.get(alert.severity, "📢")
            
            print(f"\n{icon} {alert.severity.value.upper()} ALERT: {alert.title}")
            print(f"   📅 {alert.timestamp.strftime('%H:%M:%S')}")
            print(f"   📝 {alert.message}")
            
            if alert.action_required:
                print(f"   ⚠️ ACTION REQUIRED")
            
            if alert.details:
                print(f"   📊 Details: {', '.join([f'{k}: {v}' for k, v in alert.details.items()])}")
            
            print("-" * 60)
            
            return True
        except Exception as e:
            logger.error(f"Console notification failed: {e}")
            return False
    
    def _deliver_log_file(self, alert: Alert) -> bool:
        """Deliver log file notification"""
        try:
            log_message = (f"ALERT|{alert.severity.value}|{alert.category.value}|"
                          f"{alert.timestamp.isoformat()}|{alert.title}|{alert.message}")
            
            logger.info(log_message)
            return True
        except Exception as e:
            logger.error(f"Log file notification failed: {e}")
            return False
    
    def _deliver_web_dashboard(self, alert: Alert) -> bool:
        """Deliver web dashboard notification (placeholder)"""
        # This would integrate with the live dashboard
        # For now, just return True
        return True
    
    def _deliver_email(self, alert: Alert) -> bool:
        """Deliver email notification"""
        try:
            if not self.email_config['username']:
                return False
            
            # Email delivery would be implemented here
            # For demo purposes, just log
            logger.info(f"Email notification would be sent: {alert.title}")
            return True
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return False

class AdvancedAlertSystem:
    """Main advanced alert and notification system"""
    
    def __init__(self):
        self.alert_manager = AlertManager()
        self.risk_monitor = RiskMonitor(self.alert_manager)
        self.market_monitor = MarketMonitor(self.alert_manager)
        self.performance_monitor = PerformanceMonitor(self.alert_manager)
        self.notification_delivery = NotificationDelivery()
        
        # Register alert callback for notifications
        self.alert_manager.register_alert_callback(self._on_alert_created)
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info("🚨 Advanced Alert & Notification System initialized")
    
    def _on_alert_created(self, alert: Alert) -> None:
        """Handle new alert creation"""
        # Deliver notification
        delivery_status = self.notification_delivery.deliver_notification(alert)
        
        logger.debug(f"Alert {alert.id} delivered: {delivery_status}")
    
    def start_monitoring(self) -> None:
        """Start continuous monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("🚀 Alert monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("🛑 Alert monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Clean up expired alerts
                expired_count = self.alert_manager.cleanup_expired_alerts()
                if expired_count > 0:
                    logger.debug(f"Cleaned up {expired_count} expired alerts")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)
    
    def process_trading_data(self, data: Dict[str, Any]) -> None:
        """Process trading data for alerts"""
        
        # Risk monitoring
        if 'performance_metrics' in data:
            self.risk_monitor.check_risk_metrics(data['performance_metrics'])
            self.performance_monitor.check_performance_milestones(data['performance_metrics'])
        
        # Market monitoring
        if 'market_data' in data:
            self.market_monitor.check_market_conditions(data['market_data'])
        
        # Manual alerts from insights
        if 'insights' in data:
            for insight in data['insights']:
                if insight.get('priority') == 'high':
                    self.alert_manager.create_alert(
                        severity=AlertSeverity.HIGH,
                        category=AlertCategory.TRADING_SIGNAL,
                        title=f"High Priority Insight: {insight.get('title', '')}",
                        message=insight.get('description', ''),
                        details=insight,
                        action_required=True
                    )
        
        # Anomaly alerts
        if 'anomalies' in data:
            for anomaly in data['anomalies']:
                severity_map = {
                    'critical': AlertSeverity.CRITICAL,
                    'high': AlertSeverity.HIGH,
                    'medium': AlertSeverity.MEDIUM,
                    'low': AlertSeverity.LOW
                }
                
                severity = severity_map.get(anomaly.get('severity', 'medium'), AlertSeverity.MEDIUM)
                
                self.alert_manager.create_alert(
                    severity=severity,
                    category=AlertCategory.ANOMALY,
                    title=f"Anomaly Detected: {anomaly.get('type', '')}",
                    message=anomaly.get('description', ''),
                    details=anomaly,
                    action_required=severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
                )
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert system summary"""
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            'active_alerts': len(active_alerts),
            'critical_alerts': len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            'high_alerts': len([a for a in active_alerts if a.severity == AlertSeverity.HIGH]),
            'unacknowledged_alerts': len([a for a in active_alerts if not a.acknowledged]),
            'alerts_requiring_action': len([a for a in active_alerts if a.action_required]),
            'total_alerts_generated': self.alert_manager.alert_counters['total'],
            'alert_counters': self.alert_manager.alert_counters.copy()
        }
    
    def create_custom_alert(self, severity: str, title: str, message: str, 
                           category: str = "system_health", details: Dict = None) -> str:
        """Create custom alert"""
        severity_enum = AlertSeverity(severity.lower())
        category_enum = AlertCategory(category.lower())
        
        alert = self.alert_manager.create_alert(
            severity=severity_enum,
            category=category_enum,
            title=title,
            message=message,
            details=details or {},
            action_required=severity_enum in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        )
        
        return alert.id


def main():
    """Demo of the Advanced Alert & Notification System"""
    print("\n" + "="*80)
    print("🚨 BAMBHORIA GOD-EYE V52 - ADVANCED ALERT & NOTIFICATION SYSTEM 🚨")
    print("="*80 + "\n")
    
    # Initialize alert system
    alert_system = AdvancedAlertSystem()
    
    print("🚀 Advanced Alert System initialized!")
    print("\n🎯 Capabilities:")
    print("   ✅ Real-time risk monitoring")
    print("   ✅ Market condition alerts")
    print("   ✅ Performance milestones")
    print("   ✅ Anomaly detection alerts")
    print("   ✅ Multi-channel notifications")
    print("   ✅ Intelligent prioritization")
    
    # Start monitoring
    alert_system.start_monitoring()
    
    print("\n📊 Demonstrating alert system with sample scenarios...")
    
    # Demo scenarios
    import random
    time.sleep(2)
    
    # Scenario 1: Risk alert
    print("\n🔴 SCENARIO 1: Risk threshold breach")
    alert_system.process_trading_data({
        'performance_metrics': {
            'max_drawdown': 0.08,  # 8% - exceeds 5% threshold
            'daily_pnl': -3500,
            'account_balance': 100000,
            'win_rate': 0.35  # Below 40% threshold
        }
    })
    
    time.sleep(3)
    
    # Scenario 2: Market condition alert
    print("\n🟠 SCENARIO 2: Market anomaly detected")
    alert_system.process_trading_data({
        'market_data': {
            'RELIANCE': {
                'price': 2800,
                'volume': 2500000  # High volume
            }
        }
    })
    
    time.sleep(3)
    
    # Scenario 3: Performance milestone
    print("\n🟡 SCENARIO 3: Performance milestone reached")
    alert_system.process_trading_data({
        'performance_metrics': {
            'total_pnl': 12000,  # Exceeds 10k milestone
            'trades_count': 55,  # Exceeds 50 trades milestone
            'win_rate': 0.78
        }
    })
    
    time.sleep(3)
    
    # Scenario 4: High priority insight
    print("\n🚨 SCENARIO 4: Critical trading insight")
    alert_system.process_trading_data({
        'insights': [{
            'priority': 'high',
            'title': 'Strong Buy Signal Detected',
            'description': 'Multiple indicators align for HDFC - immediate action recommended',
            'confidence': 0.92
        }]
    })
    
    time.sleep(3)
    
    # Scenario 5: Anomaly detection
    print("\n⚠️ SCENARIO 5: System anomaly detected")
    alert_system.process_trading_data({
        'anomalies': [{
            'type': 'performance_degradation',
            'severity': 'critical',
            'description': 'Strategy performance dropped significantly',
            'impact': 'Immediate review required'
        }]
    })
    
    time.sleep(2)
    
    # Get summary
    summary = alert_system.get_alert_summary()
    
    print(f"\n📋 ALERT SYSTEM SUMMARY:")
    print(f"   🚨 Critical Alerts: {summary['critical_alerts']}")
    print(f"   🔴 High Priority: {summary['high_alerts']}")
    print(f"   📢 Total Active: {summary['active_alerts']}")
    print(f"   ⚠️ Need Action: {summary['alerts_requiring_action']}")
    print(f"   📊 Total Generated: {summary['total_alerts_generated']}")
    
    print(f"\n📈 ALERT BREAKDOWN:")
    counters = summary['alert_counters']
    print(f"   Critical: {counters['critical']}")
    print(f"   High: {counters['high']}")
    print(f"   Medium: {counters['medium']}")
    print(f"   Low: {counters['low']}")
    print(f"   Info: {counters['info']}")
    
    # Test custom alert
    print(f"\n🎛️ Creating custom alert...")
    custom_alert_id = alert_system.create_custom_alert(
        severity="medium",
        title="Custom System Check",
        message="Demonstrating custom alert functionality",
        category="system_health",
        details={'demo': True, 'timestamp': time.time()}
    )
    print(f"   ✅ Custom alert created: {custom_alert_id}")
    
    time.sleep(2)
    
    print("\n" + "="*80)
    print("🎉 ADVANCED ALERT & NOTIFICATION SYSTEM - WORLD CLASS MONITORING!")
    print("✅ Intelligent Risk Monitoring")
    print("✅ Real-time Market Alerts") 
    print("✅ Performance Tracking")
    print("✅ Multi-Channel Delivery")
    print("✅ Anomaly Detection")
    print("="*80 + "\n")
    
    # Stop monitoring
    alert_system.stop_monitoring()
    
    print("Demo completed! Alert system ready for integration.")


if __name__ == "__main__":
    main()