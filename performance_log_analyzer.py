"""
performance_log_analyzer.py
Bambhoria Performance Log Analysis Tool
Analyze quantum performance monitor logs and generate insights
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
import statistics

class PerformanceLogAnalyzer:
    def __init__(self, log_file="logs/performance_monitor.json"):
        self.log_file = Path(log_file)
        self.data = []
        self.metrics_data = []
        self.alerts = []
        
    def load_logs(self):
        """Load and parse performance monitor logs"""
        if not self.log_file.exists():
            print(f"❌ Log file not found: {self.log_file}")
            return False
            
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        self.data.append(entry)
                        
                        # Separate metrics data from alerts
                        if isinstance(entry.get('msg'), dict):
                            metrics = entry['msg']
                            metrics['timestamp_full'] = entry['ts']
                            self.metrics_data.append(metrics)
                        elif isinstance(entry.get('msg'), str):
                            alert = {
                                'timestamp': entry['ts'],
                                'message': entry['msg']
                            }
                            self.alerts.append(alert)
                            
                    except json.JSONDecodeError:
                        continue
                        
            print(f"✅ Loaded {len(self.data)} log entries")
            print(f"   📊 Metrics entries: {len(self.metrics_data)}")
            print(f"   🚨 Alert entries: {len(self.alerts)}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading logs: {e}")
            return False
    
    def analyze_system_performance(self):
        """Analyze system performance metrics"""
        if not self.metrics_data:
            print("No metrics data to analyze")
            return
            
        print("\n📊 SYSTEM PERFORMANCE ANALYSIS")
        print("=" * 50)
        
        # Extract metrics
        cpu_values = [m['cpu'] for m in self.metrics_data]
        mem_values = [m['mem'] for m in self.metrics_data]
        latency_values = [m['latency_ms'] for m in self.metrics_data if m['latency_ms'] > 0]
        
        # CPU Analysis
        print(f"\n💻 CPU USAGE:")
        print(f"   Average: {statistics.mean(cpu_values):.1f}%")
        print(f"   Peak: {max(cpu_values):.1f}%")
        print(f"   Minimum: {min(cpu_values):.1f}%")
        print(f"   Std Dev: {statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0:.1f}%")
        
        # Memory Analysis
        print(f"\n🧠 MEMORY USAGE:")
        print(f"   Average: {statistics.mean(mem_values):.1f}%")
        print(f"   Peak: {max(mem_values):.1f}%")
        print(f"   Minimum: {min(mem_values):.1f}%")
        print(f"   Std Dev: {statistics.stdev(mem_values) if len(mem_values) > 1 else 0:.1f}%")
        
        # Memory pressure analysis
        memory_pressure_count = sum(1 for m in mem_values if m > 90)
        memory_pressure_pct = (memory_pressure_count / len(mem_values)) * 100
        print(f"   Memory Pressure: {memory_pressure_count}/{len(mem_values)} readings ({memory_pressure_pct:.1f}%)")
        
        # Latency Analysis
        if latency_values:
            print(f"\n🌐 NETWORK LATENCY:")
            print(f"   Average: {statistics.mean(latency_values):.1f}ms")
            print(f"   Peak: {max(latency_values):.1f}ms")
            print(f"   Minimum: {min(latency_values):.1f}ms")
        else:
            print(f"\n🌐 NETWORK LATENCY: No latency data (services not responding)")
        
        # Performance Trends
        print(f"\n📈 PERFORMANCE TRENDS:")
        if len(cpu_values) >= 2:
            cpu_trend = "Increasing" if cpu_values[-1] > cpu_values[0] else "Decreasing"
            mem_trend = "Increasing" if mem_values[-1] > mem_values[0] else "Decreasing"
            print(f"   CPU Trend: {cpu_trend}")
            print(f"   Memory Trend: {mem_trend}")
            
        # System Health Score
        avg_cpu = statistics.mean(cpu_values)
        avg_mem = statistics.mean(mem_values)
        
        health_score = 100
        health_score -= max(0, (avg_cpu - 50) * 0.8)  # Penalize high CPU
        health_score -= max(0, (avg_mem - 60) * 0.6)  # Penalize high memory
        health_score = max(0, min(100, health_score))
        
        print(f"\n🎯 OVERALL HEALTH SCORE: {health_score:.1f}/100")
        
        if health_score >= 80:
            print("   Status: ✅ EXCELLENT")
        elif health_score >= 60:
            print("   Status: 🟡 GOOD")
        elif health_score >= 40:
            print("   Status: 🟠 FAIR") 
        else:
            print("   Status: 🔴 POOR")
    
    def analyze_alerts(self):
        """Analyze system alerts and warnings"""
        if not self.alerts:
            print("\n🚨 No alerts recorded")
            return
            
        print(f"\n🚨 ALERT ANALYSIS")
        print("=" * 30)
        
        # Count alert types
        alert_types = {}
        for alert in self.alerts:
            message = alert['message']
            if "Memory pressure" in message:
                alert_types['Memory Pressure'] = alert_types.get('Memory Pressure', 0) + 1
            elif "High CPU" in message:
                alert_types['High CPU'] = alert_types.get('High CPU', 0) + 1
            elif "High latency" in message:
                alert_types['High Latency'] = alert_types.get('High Latency', 0) + 1
            elif "started" in message:
                alert_types['System Events'] = alert_types.get('System Events', 0) + 1
            elif "stopped" in message:
                alert_types['System Events'] = alert_types.get('System Events', 0) + 1
            else:
                alert_types['Other'] = alert_types.get('Other', 0) + 1
        
        print(f"Total Alerts: {len(self.alerts)}")
        for alert_type, count in alert_types.items():
            print(f"   {alert_type}: {count}")
        
        # Recent alerts
        print(f"\n🕐 RECENT ALERTS:")
        recent_alerts = self.alerts[-5:] if len(self.alerts) > 5 else self.alerts
        for alert in recent_alerts:
            timestamp = datetime.fromisoformat(alert['timestamp']).strftime("%H:%M:%S")
            print(f"   [{timestamp}] {alert['message']}")
    
    def generate_timeline(self):
        """Generate performance timeline"""
        if not self.metrics_data:
            return
            
        print(f"\n⏰ PERFORMANCE TIMELINE")
        print("=" * 40)
        
        for i, metric in enumerate(self.metrics_data):
            timestamp = metric['timestamp']
            cpu = metric['cpu']
            mem = metric['mem']
            uptime = metric['uptime_min']
            
            # Status indicators
            cpu_status = "🔴" if cpu > 80 else "🟡" if cpu > 60 else "🟢"
            mem_status = "🔴" if mem > 90 else "🟡" if mem > 80 else "🟢"
            
            print(f"   [{timestamp}] CPU: {cpu_status}{cpu:5.1f}% | MEM: {mem_status}{mem:5.1f}% | Uptime: {uptime:.2f}m")
    
    def generate_recommendations(self):
        """Generate system optimization recommendations"""
        if not self.metrics_data:
            return
            
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS")
        print("=" * 45)
        
        cpu_values = [m['cpu'] for m in self.metrics_data]
        mem_values = [m['mem'] for m in self.metrics_data]
        
        avg_cpu = statistics.mean(cpu_values)
        avg_mem = statistics.mean(mem_values)
        peak_mem = max(mem_values)
        
        recommendations = []
        
        # CPU recommendations
        if avg_cpu > 70:
            recommendations.append("🔧 High CPU usage detected. Consider optimizing trading algorithms or adding CPU resources.")
        elif avg_cpu < 20:
            recommendations.append("💡 Low CPU usage. System has capacity for additional workloads.")
            
        # Memory recommendations  
        if peak_mem > 95:
            recommendations.append("🚨 CRITICAL: Memory usage extremely high. Immediate action required!")
        elif avg_mem > 85:
            recommendations.append("⚠️  High memory usage. Consider increasing system RAM or optimizing memory usage.")
        elif peak_mem > 90:
            recommendations.append("🔍 Memory spikes detected. Monitor for memory leaks in trading components.")
            
        # Alert frequency recommendations
        memory_alerts = len([a for a in self.alerts if "Memory pressure" in a['message']])
        if memory_alerts > len(self.metrics_data) * 0.5:
            recommendations.append("📊 Frequent memory alerts. Consider adjusting memory thresholds or system optimization.")
            
        # General recommendations
        if len(self.metrics_data) < 10:
            recommendations.append("📈 Insufficient monitoring data. Run system longer for better analysis.")
        
        if not recommendations:
            recommendations.append("✅ System performance is optimal. No immediate recommendations.")
            
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    def export_summary_report(self, output_file="logs/performance_analysis_report.txt"):
        """Export detailed analysis report"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("BAMBHORIA QUANTUM PERFORMANCE MONITOR - ANALYSIS REPORT\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Log File: {self.log_file}\n")
                f.write(f"Total Entries: {len(self.data)}\n")
                f.write(f"Metrics Entries: {len(self.metrics_data)}\n")
                f.write(f"Alert Entries: {len(self.alerts)}\n\n")
                
                if self.metrics_data:
                    cpu_values = [m['cpu'] for m in self.metrics_data]
                    mem_values = [m['mem'] for m in self.metrics_data]
                    
                    f.write("SYSTEM PERFORMANCE SUMMARY\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"CPU Usage - Avg: {statistics.mean(cpu_values):.1f}%, Peak: {max(cpu_values):.1f}%\n")
                    f.write(f"Memory Usage - Avg: {statistics.mean(mem_values):.1f}%, Peak: {max(mem_values):.1f}%\n")
                    
                    memory_pressure_count = sum(1 for m in mem_values if m > 90)
                    f.write(f"Memory Pressure Events: {memory_pressure_count}\n\n")
                
                if self.alerts:
                    f.write("ALERTS SUMMARY\n")
                    f.write("-" * 15 + "\n")
                    f.write(f"Total Alerts: {len(self.alerts)}\n")
                    for alert in self.alerts[-10:]:  # Last 10 alerts
                        timestamp = datetime.fromisoformat(alert['timestamp']).strftime("%H:%M:%S")
                        f.write(f"[{timestamp}] {alert['message']}\n")
            
            print(f"\n📄 Analysis report exported to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error exporting report: {e}")
    
    def run_complete_analysis(self):
        """Run complete performance analysis"""
        print("🎯 BAMBHORIA QUANTUM PERFORMANCE LOG ANALYZER")
        print("=" * 55)
        
        if not self.load_logs():
            return
            
        self.analyze_system_performance()
        self.analyze_alerts()
        self.generate_timeline()
        self.generate_recommendations()
        self.export_summary_report()
        
        print(f"\n✅ Analysis complete!")

def main():
    analyzer = PerformanceLogAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()