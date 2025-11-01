# Bambhoria Auto-Heal Engine v1.0

## 🩺 Overview
The Auto-Heal Engine is a robust monitoring and recovery system that automatically detects and restarts crashed services in the Bambhoria trading platform.

## 🎯 Features
- **Real-time Process Monitoring**: Continuously monitors critical services
- **Automatic Recovery**: Restarts crashed services automatically
- **Health Check Integration**: Uses `/api/health` endpoints for service validation
- **Comprehensive Logging**: Maintains detailed recovery logs in JSON format
- **Console Management**: Spawns services in separate console windows for better monitoring

## 🔧 Monitored Services
1. **Dashboard Server** (`dashboard_server.py`)
2. **Mock Live Feed** (`mock_live_feed.py`) 
3. **Signal Generator** (`signal_generator.py`)

## 📊 Configuration
```python
SERVICES = {
    "dashboard_server.py": "python dashboard/dashboard_server.py",
    "mock_live_feed.py":   "python tools/mock_live_feed.py", 
    "signal_generator.py": "python core_engine/signal_generator.py",
}
HEALTH_URL  = "http://localhost:5000/api/health"   # dashboard health check
CHECK_DELAY = 10                                   # seconds between checks
LOG_PATH    = Path("logs/auto_heal_log.json")      # recovery log location
```

## 🚀 Usage

### Start Auto-Heal Engine
```bash
python auto_heal_engine.py
```

### Test Auto-Heal Functionality  
```bash
python auto_heal_test.py
```

## 📋 Log Format
Recovery logs are stored in `logs/auto_heal_log.json`:
```json
{"time": "2025-10-26 16:30:15", "msg": "🩺  Auto-Heal Engine started."}
{"time": "2025-10-26 16:30:25", "msg": "✅  dashboard_server.py OK"}
{"time": "2025-10-26 16:30:35", "msg": "⚠️  mock_live_feed.py not running → Restarting..."}
```

## 🔍 Health Monitoring
- **Process Detection**: Uses `psutil` to scan running processes
- **Health Endpoint**: Pings `/api/health` for dashboard responsiveness
- **Recovery Actions**: Automatically restarts failed services in new console windows

## 🛡️ Recovery Behavior
1. **Process Check**: Scans for service processes every 10 seconds
2. **Health Ping**: Tests dashboard responsiveness via HTTP health check
3. **Auto-Restart**: Launches crashed services in new console windows
4. **Logging**: Records all recovery actions with timestamps

## 🎮 Manual Testing
1. Start the auto-heal engine
2. Manually close any monitored service window
3. Watch the auto-heal engine detect and restart the service
4. Check logs for recovery actions

## 🔧 Integration with Health Endpoints
The auto-heal engine leverages the health endpoints implemented across all dashboard services:
- `http://localhost:5000/api/health` (Main Dashboard)
- `http://localhost:5004/api/health` (Pipeline Dashboard) 
- `http://localhost:5008/api/health` (System Monitor)
- `http://localhost:5009/api/health` (Performance Dashboard)

## 💡 Best Practices
- Run auto-heal engine in a dedicated console window
- Monitor logs regularly for system health insights
- Adjust `CHECK_DELAY` based on system requirements
- Customize `SERVICES` dictionary for specific service configurations

## 🚨 Error Handling
- Graceful exception handling for process monitoring
- Timeout protection for health checks
- Automatic log directory creation
- Safe process restart procedures

## 🎯 Production Deployment
For production use:
1. Run as a Windows service or Linux daemon
2. Implement email/SMS alerts for critical failures
3. Add service dependency management
4. Configure appropriate monitoring intervals
5. Set up log rotation for long-term operation