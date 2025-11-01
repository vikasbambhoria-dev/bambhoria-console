"""
performance_log_viewer.py
Real-time Performance Log Viewer for Bambhoria Quantum Monitor
Live tail and analysis of performance logs
"""

import json
import time
import os
from datetime import datetime
from pathlib import Path
import threading

class PerformanceLogViewer:
    def __init__(self, log_file="logs/performance_monitor.json"):
        self.log_file = Path(log_file)
        self.last_position = 0
        self.running = False
        self.stats = {
            "total_entries": 0,
            "alerts": 0,
            "last_cpu": 0,
            "last_memory": 0,
            "session_start": datetime.now()
        }
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print monitoring header"""
        print("=" * 80)
        print("🎯 BAMBHORIA QUANTUM PERFORMANCE MONITOR - LIVE VIEW")
        print("=" * 80)
        print(f"📂 Log File: {self.log_file}")
        print(f"⏰ Session Start: {self.stats['session_start'].strftime('%H:%M:%S')}")
        print(f"📊 Total Entries: {self.stats['total_entries']} | 🚨 Alerts: {self.stats['alerts']}")
        
        if self.stats['last_cpu'] > 0:
            cpu_status = "🔴" if self.stats['last_cpu'] > 80 else "🟡" if self.stats['last_cpu'] > 60 else "🟢"
            mem_status = "🔴" if self.stats['last_memory'] > 90 else "🟡" if self.stats['last_memory'] > 80 else "🟢"
            print(f"💻 Latest: CPU {cpu_status}{self.stats['last_cpu']:.1f}% | Memory {mem_status}{self.stats['last_memory']:.1f}%")
        
        print("-" * 80)
    
    def format_log_entry(self, entry):
        """Format log entry for display"""
        timestamp = datetime.fromisoformat(entry['ts']).strftime("%H:%M:%S")
        msg = entry['msg']
        
        if isinstance(msg, dict):
            # Metrics entry
            cpu = msg.get('cpu', 0)
            mem = msg.get('mem', 0)
            latency = msg.get('latency_ms', 0)
            uptime = msg.get('uptime_min', 0)
            
            # Update stats
            self.stats['last_cpu'] = cpu
            self.stats['last_memory'] = mem
            
            # Color coding
            cpu_color = "🔴" if cpu > 80 else "🟡" if cpu > 60 else "🟢"
            mem_color = "🔴" if mem > 90 else "🟡" if mem > 80 else "🟢"
            
            return (f"[{timestamp}] 📊 METRICS: "
                   f"CPU {cpu_color}{cpu:5.1f}% | "
                   f"MEM {mem_color}{mem:5.1f}% | "
                   f"Latency {latency:4.0f}ms | "
                   f"Uptime {uptime:6.2f}m")
        
        elif isinstance(msg, str):
            # Alert/message entry
            if "Memory pressure" in msg:
                self.stats['alerts'] += 1
                return f"[{timestamp}] 🚨 ALERT: {msg}"
            elif "started" in msg:
                return f"[{timestamp}] 🚀 EVENT: {msg}"
            elif "stopped" in msg:
                return f"[{timestamp}] 🛑 EVENT: {msg}"
            elif "High CPU" in msg:
                self.stats['alerts'] += 1
                return f"[{timestamp}] 🔴 CRITICAL: {msg}"
            else:
                return f"[{timestamp}] ℹ️  INFO: {msg}"
        
        return f"[{timestamp}] {msg}"
    
    def tail_log_file(self):
        """Tail the log file for new entries"""
        if not self.log_file.exists():
            print(f"❌ Log file not found: {self.log_file}")
            return
        
        # Get initial file size
        file_size = self.log_file.stat().st_size
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            # Move to end of file
            f.seek(file_size)
            
            while self.running:
                line = f.readline()
                if line:
                    try:
                        entry = json.loads(line.strip())
                        self.stats['total_entries'] += 1
                        formatted = self.format_log_entry(entry)
                        print(formatted)
                    except json.JSONDecodeError:
                        pass
                else:
                    time.sleep(0.5)  # Wait for new data
    
    def monitor_performance_live(self):
        """Start live performance monitoring"""
        self.running = True
        
        # Clear screen and show header
        self.clear_screen()
        self.print_header()
        
        # Load existing log entries first
        if self.log_file.exists():
            print("📖 Loading existing log entries...")
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        self.stats['total_entries'] += 1
                        formatted = self.format_log_entry(entry)
                        print(formatted)
                    except json.JSONDecodeError:
                        pass
        
        print("\n🔄 Monitoring for new entries (Press Ctrl+C to stop)...")
        print("-" * 80)
        
        try:
            # Start tailing new entries
            self.tail_log_file()
        except KeyboardInterrupt:
            self.running = False
            print("\n🛑 Live monitoring stopped")
    
    def show_log_summary(self):
        """Show summary of current log file"""
        if not self.log_file.exists():
            print(f"❌ Log file not found: {self.log_file}")
            return
        
        print("📊 PERFORMANCE LOG SUMMARY")
        print("=" * 40)
        
        entries = []
        metrics = []
        alerts = []
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                    
                    if isinstance(entry.get('msg'), dict):
                        metrics.append(entry['msg'])
                    elif isinstance(entry.get('msg'), str) and any(keyword in entry['msg'] for keyword in ['pressure', 'High', 'CRITICAL']):
                        alerts.append(entry)
                        
                except json.JSONDecodeError:
                    pass
        
        print(f"Total Entries: {len(entries)}")
        print(f"Metrics Entries: {len(metrics)}")
        print(f"Alert Entries: {len(alerts)}")
        
        if metrics:
            cpu_values = [m['cpu'] for m in metrics]
            mem_values = [m['mem'] for m in metrics]
            
            print(f"\nCPU: Avg {sum(cpu_values)/len(cpu_values):.1f}%, Peak {max(cpu_values):.1f}%")
            print(f"Memory: Avg {sum(mem_values)/len(mem_values):.1f}%, Peak {max(mem_values):.1f}%")
        
        if alerts:
            print(f"\nRecent Alerts:")
            for alert in alerts[-3:]:
                timestamp = datetime.fromisoformat(alert['ts']).strftime("%H:%M:%S")
                print(f"  [{timestamp}] {alert['msg']}")

def main():
    import sys
    
    viewer = PerformanceLogViewer()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        viewer.show_log_summary()
    else:
        viewer.monitor_performance_live()

if __name__ == "__main__":
    main()