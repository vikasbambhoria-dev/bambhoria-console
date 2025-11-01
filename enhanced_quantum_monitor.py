"""
enhanced_quantum_monitor.py
Enhanced Bambhoria Quantum Performance Monitor v2.0
Author: Vikas Bambhoria
Advanced monitoring with integration to trading system components
"""

import os, psutil, time, json, requests, statistics, threading
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import subprocess
import socket

class QuantumPerformanceMonitor:
    def __init__(self):
        self.monitor_log = Path("logs/performance_monitor.json")
        self.ping_interval = 5
        self.latency_samples = deque(maxlen=100)
        self.tick_rate = deque(maxlen=50)
        self.trades_success = 0
        self.trades_failed = 0
        self.start_time = datetime.now()
        
        # Multiple endpoints to monitor
        self.endpoints = {
            "dashboard": "http://localhost:5006/api/pnl",
            "system_monitor": "http://localhost:5008/api/system_status",
            "analytics": "http://localhost:5000/api/status",
            "pipeline": "http://localhost:5005/api/status"
        }
        
        # System thresholds
        self.thresholds = {
            "cpu_warning": 75,
            "cpu_critical": 85,
            "memory_warning": 80,
            "memory_critical": 90,
            "latency_warning": 500,
            "latency_critical": 800,
            "disk_warning": 85,
            "disk_critical": 95
        }
        
        os.makedirs(self.monitor_log.parent, exist_ok=True)
        self._log("🚀 Enhanced Quantum Performance Monitor v2.0 initialized")
    
    def _log(self, msg):
        """Enhanced logging with console and file output"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {msg}"
        print(log_entry)
        
        # Write to log file
        with open(self.monitor_log, "a", encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "message": str(msg)
            }) + "\n")
    
    def check_port_status(self, port):
        """Check if a port is open/listening"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                return result == 0
        except:
            return False
    
    def get_endpoint_latency(self, name, url):
        """Measure latency for specific endpoint"""
        t0 = time.time()
        try:
            resp = requests.get(url, timeout=3)
            latency = (time.time() - t0) * 1000
            
            if resp.status_code == 200:
                self.latency_samples.append(latency)
                return {"status": "OK", "latency": round(latency, 2), "code": resp.status_code}
            else:
                return {"status": "ERROR", "latency": None, "code": resp.status_code}
        except requests.exceptions.Timeout:
            return {"status": "TIMEOUT", "latency": None, "code": None}
        except requests.exceptions.ConnectionError:
            return {"status": "DOWN", "latency": None, "code": None}
        except Exception as e:
            return {"status": "ERROR", "latency": None, "code": None, "error": str(e)}
    
    def get_system_metrics(self):
        """Comprehensive system metrics collection"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "frequency": cpu_freq.current if cpu_freq else None
                },
                "memory": {
                    "percent": memory.percent,
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "total_gb": round(memory.total / (1024**3), 2)
                },
                "swap": {
                    "percent": swap.percent,
                    "used_gb": round(swap.used / (1024**3), 2)
                },
                "disk": {
                    "percent": round((disk.used / disk.total) * 100, 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2)
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "processes": process_count
            }
        except Exception as e:
            self._log(f"❌ Error collecting system metrics: {e}")
            return {}
    
    def check_trading_system_health(self):
        """Check health of trading system components"""
        health_status = {}
        
        # Check critical ports
        critical_ports = {
            "System Monitor": 5008,
            "Analytics Dashboard": 5006,
            "Pipeline Monitor": 5005,
            "Main Dashboard": 5000
        }
        
        for name, port in critical_ports.items():
            is_up = self.check_port_status(port)
            health_status[name] = {
                "port": port,
                "status": "UP" if is_up else "DOWN",
                "responsive": is_up
            }
        
        # Check endpoints
        for name, url in self.endpoints.items():
            result = self.get_endpoint_latency(name, url)
            health_status[f"{name}_endpoint"] = result
        
        return health_status
    
    def calculate_performance_score(self, metrics):
        """Calculate overall system performance score (0-100)"""
        try:
            score = 100
            
            # CPU impact (30% weight)
            cpu_penalty = max(0, (metrics.get("cpu", {}).get("percent", 0) - 50) * 0.6)
            score -= cpu_penalty
            
            # Memory impact (25% weight)
            mem_penalty = max(0, (metrics.get("memory", {}).get("percent", 0) - 60) * 0.5)
            score -= mem_penalty
            
            # Disk impact (15% weight)
            disk_penalty = max(0, (metrics.get("disk", {}).get("percent", 0) - 70) * 0.3)
            score -= disk_penalty
            
            # Latency impact (20% weight)
            avg_latency = statistics.mean(self.latency_samples) if self.latency_samples else 0
            latency_penalty = max(0, (avg_latency - 200) * 0.1)
            score -= latency_penalty
            
            # Service availability (10% weight)
            # This would be calculated based on endpoint health
            
            return max(0, min(100, round(score, 1)))
        except:
            return 50  # Default neutral score
    
    def generate_alerts(self, metrics, health_status):
        """Generate system alerts based on thresholds"""
        alerts = []
        
        # CPU alerts
        cpu_percent = metrics.get("cpu", {}).get("percent", 0)
        if cpu_percent > self.thresholds["cpu_critical"]:
            alerts.append(f"🔴 CRITICAL: CPU usage at {cpu_percent}%")
        elif cpu_percent > self.thresholds["cpu_warning"]:
            alerts.append(f"🟡 WARNING: CPU usage at {cpu_percent}%")
        
        # Memory alerts
        mem_percent = metrics.get("memory", {}).get("percent", 0)
        if mem_percent > self.thresholds["memory_critical"]:
            alerts.append(f"🔴 CRITICAL: Memory usage at {mem_percent}%")
        elif mem_percent > self.thresholds["memory_warning"]:
            alerts.append(f"🟡 WARNING: Memory usage at {mem_percent}%")
        
        # Disk alerts
        disk_percent = metrics.get("disk", {}).get("percent", 0)
        if disk_percent > self.thresholds["disk_critical"]:
            alerts.append(f"🔴 CRITICAL: Disk usage at {disk_percent}%")
        elif disk_percent > self.thresholds["disk_warning"]:
            alerts.append(f"🟡 WARNING: Disk usage at {disk_percent}%")
        
        # Latency alerts
        if self.latency_samples:
            avg_latency = statistics.mean(self.latency_samples)
            if avg_latency > self.thresholds["latency_critical"]:
                alerts.append(f"🔴 CRITICAL: High latency {avg_latency:.1f}ms")
            elif avg_latency > self.thresholds["latency_warning"]:
                alerts.append(f"🟡 WARNING: Elevated latency {avg_latency:.1f}ms")
        
        # Service alerts
        for service, status in health_status.items():
            if isinstance(status, dict) and status.get("status") == "DOWN":
                alerts.append(f"🔴 CRITICAL: {service} is DOWN")
            elif isinstance(status, dict) and status.get("status") == "TIMEOUT":
                alerts.append(f"🟡 WARNING: {service} is not responding")
        
        return alerts
    
    def monitor_system(self):
        """Main monitoring loop"""
        self._log("🚀 Quantum Performance Monitor started")
        
        try:
            while True:
                # Collect system metrics
                metrics = self.get_system_metrics()
                
                # Check trading system health
                health_status = self.check_trading_system_health()
                
                # Calculate performance score
                performance_score = self.calculate_performance_score(metrics)
                
                # Calculate statistics
                uptime_minutes = (datetime.now() - self.start_time).total_seconds() / 60
                avg_latency = round(statistics.mean(self.latency_samples), 2) if self.latency_samples else 0
                tick_rate = round(statistics.mean(self.tick_rate), 2) if len(self.tick_rate) >= 5 else 0
                trade_total = self.trades_success + self.trades_failed
                trade_success_rate = round((self.trades_success / trade_total) * 100, 2) if trade_total else 100
                
                # Generate comprehensive status
                status_report = {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "performance_score": performance_score,
                    "uptime_minutes": round(uptime_minutes, 2),
                    "system_metrics": metrics,
                    "trading_health": health_status,
                    "statistics": {
                        "avg_latency_ms": avg_latency,
                        "tick_rate": tick_rate,
                        "trade_success_rate": trade_success_rate,
                        "total_trades": trade_total
                    }
                }
                
                # Generate and log alerts
                alerts = self.generate_alerts(metrics, health_status)
                if alerts:
                    for alert in alerts:
                        self._log(alert)
                
                # Log status summary
                cpu = metrics.get("cpu", {}).get("percent", 0)
                mem = metrics.get("memory", {}).get("percent", 0)
                disk = metrics.get("disk", {}).get("percent", 0)
                
                summary = (f"💎 Score: {performance_score} | "
                          f"CPU: {cpu}% | "
                          f"MEM: {mem}% | "
                          f"DISK: {disk}% | "
                          f"Latency: {avg_latency}ms | "
                          f"Uptime: {uptime_minutes:.1f}m")
                
                self._log(summary)
                
                # Save detailed report to log
                with open(self.monitor_log.parent / "detailed_performance.json", "a", encoding='utf-8') as f:
                    f.write(json.dumps(status_report, indent=2) + "\n")
                
                time.sleep(self.ping_interval)
                
        except KeyboardInterrupt:
            self._log("🛑 Monitor stopped by user")
        except Exception as e:
            self._log(f"❌ Monitor error: {e}")

def main():
    """Main entry point"""
    monitor = QuantumPerformanceMonitor()
    monitor.monitor_system()

if __name__ == "__main__":
    main()