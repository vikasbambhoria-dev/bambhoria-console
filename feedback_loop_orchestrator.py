"""
feedback_loop_orchestrator.py
Bambhoria Feedback Loop Orchestrator v1.0
Author: Vikas Bambhoria
Purpose:
 - Orchestrates the complete feedback loop: Trade → Log → Neural → Pattern → Weight Update
 - Monitors trading performance and triggers neural retraining
 - Provides real-time analytics on system learning progress
"""

import time, json, os, threading, schedule, pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from neural_insight_engine import train_pattern_model, load_trades, analyse_trends
from adaptive_signal_engine import analyze_neural_patterns, update_neural_patterns

# ---------- CONFIG ----------
FEEDBACK_LOG = Path("logs/feedback_loop_log.json")
PERFORMANCE_LOG = Path("logs/performance_metrics.json")
LOOP_INTERVAL_MINUTES = 30  # How often to run feedback loop
RETRAIN_THRESHOLD = 20      # Minimum new trades before retraining

os.makedirs(FEEDBACK_LOG.parent, exist_ok=True)
os.makedirs(PERFORMANCE_LOG.parent, exist_ok=True)

# ---------- FEEDBACK LOOP MONITOR ----------
class FeedbackLoopOrchestrator:
    def __init__(self):
        self.last_trade_count = 0
        self.last_retrain_time = datetime.now()
        self.loop_count = 0
        self.performance_history = []
        
    def log_feedback_event(self, event_type, data):
        """Log feedback loop events"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data,
            "loop_count": self.loop_count
        }
        
        with open(FEEDBACK_LOG, "a") as f:
            f.write(json.dumps(event) + "\n")
        
        print(f"🔄 [{event_type}] {data.get('message', '')}")
    
    def analyze_system_performance(self):
        """Analyze overall system performance and learning progress"""
        df = load_trades()
        
        if df.empty:
            return {"status": "no_data", "message": "No trades to analyze"}
        
        # Calculate performance metrics
        total_trades = len(df)
        profitable_trades = (df['pnl'] > 0).sum()
        win_rate = float(profitable_trades / total_trades) if total_trades > 0 else 0.0
        total_pnl = float(df['pnl'].sum())
        avg_pnl = float(df['pnl'].mean())
        
        # Recent performance (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        df['datetime'] = pd.to_datetime(df['time'])
        recent_trades = df[df['datetime'] > recent_cutoff]
        
        recent_win_rate = float((recent_trades['pnl'] > 0).mean()) if not recent_trades.empty else 0.0
        recent_pnl = float(recent_trades['pnl'].sum()) if not recent_trades.empty else 0.0
        
        recent_metrics = {
            "recent_trades": len(recent_trades),
            "recent_win_rate": recent_win_rate,
            "recent_pnl": recent_pnl
        }
        
        # Learning progress indicators
        new_trades_since_retrain = total_trades - self.last_trade_count
        time_since_retrain = (datetime.now() - self.last_retrain_time).total_seconds() / 3600
        
        performance_data = {
            "timestamp": datetime.now().isoformat(),
            "total_trades": int(total_trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl_per_trade": avg_pnl,
            "new_trades_since_retrain": int(new_trades_since_retrain),
            "hours_since_retrain": float(time_since_retrain),
            **recent_metrics
        }
        
        # Save performance metrics
        with open(PERFORMANCE_LOG, "a") as f:
            f.write(json.dumps(performance_data) + "\n")
        
        self.performance_history.append(performance_data)
        return performance_data
    
    def should_retrain(self, performance_data):
        """Determine if neural model should be retrained"""
        new_trades = performance_data["new_trades_since_retrain"]
        hours_since = performance_data["hours_since_retrain"]
        
        # Retrain conditions
        enough_new_trades = new_trades >= RETRAIN_THRESHOLD
        enough_time_passed = hours_since >= 2  # At least 2 hours since last retrain
        
        # Performance-based triggers
        recent_win_rate = performance_data.get("recent_win_rate", 0)
        declining_performance = recent_win_rate < 0.4 and performance_data["recent_trades"] > 5
        
        return {
            "should_retrain": enough_new_trades and enough_time_passed,
            "emergency_retrain": declining_performance,
            "reasons": {
                "new_trades": new_trades >= RETRAIN_THRESHOLD,
                "time_passed": enough_time_passed,
                "declining_performance": declining_performance
            }
        }
    
    def execute_feedback_loop(self):
        """Execute one complete feedback loop cycle"""
        self.loop_count += 1
        
        self.log_feedback_event("LOOP_START", {
            "message": f"Starting feedback loop cycle #{self.loop_count}",
            "loop_count": self.loop_count
        })
        
        try:
            # 1. Analyze current performance
            performance = self.analyze_system_performance()
            
            self.log_feedback_event("PERFORMANCE_ANALYSIS", {
                "message": f"Win rate: {performance['win_rate']:.1%}, Total PnL: ₹{performance['total_pnl']:+.1f}",
                "performance": performance
            })
            
            # 2. Check if retraining is needed
            retrain_decision = self.should_retrain(performance)
            
            if retrain_decision["should_retrain"] or retrain_decision["emergency_retrain"]:
                self.log_feedback_event("RETRAIN_TRIGGERED", {
                    "message": "Neural model retraining triggered",
                    "reasons": retrain_decision["reasons"],
                    "emergency": retrain_decision["emergency_retrain"]
                })
                
                # 3. Execute neural retraining
                try:
                    train_pattern_model()
                    update_neural_patterns()
                    
                    self.last_trade_count = performance["total_trades"]
                    self.last_retrain_time = datetime.now()
                    
                    self.log_feedback_event("RETRAIN_COMPLETE", {
                        "message": "Neural model successfully retrained",
                        "new_trade_count": self.last_trade_count
                    })
                    
                except Exception as e:
                    self.log_feedback_event("RETRAIN_ERROR", {
                        "message": f"Neural retraining failed: {str(e)}",
                        "error": str(e)
                    })
            
            else:
                self.log_feedback_event("RETRAIN_SKIPPED", {
                    "message": "Retraining not needed",
                    "reasons": retrain_decision["reasons"]
                })
            
            # 4. Analyze neural patterns
            analyze_neural_patterns()
            
            # 5. Generate insights report
            self.generate_insights_report(performance)
            
            self.log_feedback_event("LOOP_COMPLETE", {
                "message": f"Feedback loop cycle #{self.loop_count} completed successfully"
            })
            
        except Exception as e:
            self.log_feedback_event("LOOP_ERROR", {
                "message": f"Feedback loop failed: {str(e)}",
                "error": str(e)
            })
    
    def generate_insights_report(self, performance):
        """Generate actionable insights from feedback loop analysis"""
        insights = []
        
        # Performance insights
        if performance["win_rate"] > 0.6:
            insights.append("🎯 Strong win rate - system learning effectively")
        elif performance["win_rate"] < 0.4:
            insights.append("⚠️ Low win rate - consider strategy adjustments")
        
        # PnL insights
        if performance["total_pnl"] > 0:
            insights.append(f"💰 Profitable system: ₹{performance['total_pnl']:+.1f} total PnL")
        else:
            insights.append(f"🔴 System in loss: ₹{performance['total_pnl']:+.1f} - review strategy")
        
        # Recent performance
        recent_pnl = performance.get("recent_pnl", 0)
        if recent_pnl > 0:
            insights.append("📈 Recent performance positive")
        elif recent_pnl < -100:
            insights.append("📉 Recent performance concerning - consider pause")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "insights": insights,
            "performance_summary": performance,
            "loop_count": self.loop_count
        }
        
        self.log_feedback_event("INSIGHTS_GENERATED", {
            "message": f"Generated {len(insights)} insights",
            "insights": insights
        })
        
        return report
    
    def start_continuous_loop(self):
        """Start continuous feedback loop monitoring"""
        print("🔄 Starting Bambhoria Feedback Loop Orchestrator")
        print(f"📊 Loop interval: {LOOP_INTERVAL_MINUTES} minutes")
        print(f"🧠 Retrain threshold: {RETRAIN_THRESHOLD} new trades")
        print("─" * 60)
        
        # Schedule periodic execution
        schedule.every(LOOP_INTERVAL_MINUTES).minutes.do(self.execute_feedback_loop)
        
        # Run initial feedback loop
        self.execute_feedback_loop()
        
        # Continuous monitoring
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                print("\n🛑 Feedback loop orchestrator stopped")
                break
            except Exception as e:
                print(f"❌ Orchestrator error: {e}")
                time.sleep(300)  # Wait 5 minutes before retry
    
    def get_status_summary(self):
        """Get current status summary"""
        performance = self.analyze_system_performance()
        
        summary = {
            "orchestrator_status": "running",
            "loop_count": self.loop_count,
            "last_loop": datetime.now().isoformat(),
            "performance": performance,
            "next_retrain_eligibility": {
                "trades_needed": max(0, RETRAIN_THRESHOLD - performance.get("new_trades_since_retrain", 0)),
                "time_passed_hours": performance.get("hours_since_retrain", 0)
            }
        }
        
        return summary

# ---------- STANDALONE FUNCTIONS ----------
def run_single_feedback_cycle():
    """Run a single feedback loop cycle (for testing)"""
    orchestrator = FeedbackLoopOrchestrator()
    orchestrator.execute_feedback_loop()
    return orchestrator.get_status_summary()

def monitor_feedback_logs(lines=20):
    """Monitor recent feedback loop logs"""
    if not FEEDBACK_LOG.exists():
        print("No feedback logs found")
        return
    
    print(f"📋 Recent {lines} feedback loop events:")
    print("─" * 50)
    
    with open(FEEDBACK_LOG, "r") as f:
        logs = f.readlines()
    
    for log_line in logs[-lines:]:
        event = json.loads(log_line.strip())
        timestamp = event["timestamp"][:19]  # Remove microseconds
        event_type = event["event_type"]
        message = event["data"].get("message", "")
        print(f"[{timestamp}] {event_type}: {message}")

# ---------- MAIN ----------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "single":
            print("🔄 Running single feedback loop cycle...")
            summary = run_single_feedback_cycle()
            print("\n📊 Status Summary:")
            print(json.dumps(summary, indent=2, default=str))
        
        elif sys.argv[1] == "logs":
            monitor_feedback_logs()
        
        elif sys.argv[1] == "status":
            orchestrator = FeedbackLoopOrchestrator()
            summary = orchestrator.get_status_summary()
            print("📊 Current Status:")
            print(json.dumps(summary, indent=2, default=str))
    
    else:
        # Start continuous monitoring
        orchestrator = FeedbackLoopOrchestrator()
        orchestrator.start_continuous_loop()