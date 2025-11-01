# launch_complete_system.ps1
# Bambhoria Complete Trading System PowerShell Launcher
# Starts all components with proper coordination

Write-Host "🎯 Bambhoria Complete Trading System Launcher" -ForegroundColor Green
Write-Host "=" * 60

# Change to the project directory
Set-Location "d:\bambhoria\godeye_v50_plus_auto_full_do_best"

$processes = @()

try {
    Write-Host "🚀 Starting System Monitor Dashboard..." -ForegroundColor Yellow
    $monitor = Start-Process python -ArgumentList "system_monitor_dashboard.py" -PassThru
    $processes += @{Name="Monitor Dashboard"; Process=$monitor}
    Start-Sleep 3
    
    Write-Host "🚀 Starting Analytics Dashboard..." -ForegroundColor Yellow
    $analytics = Start-Process python -ArgumentList "dashboard_server.py" -PassThru
    $processes += @{Name="Analytics Dashboard"; Process=$analytics}
    Start-Sleep 3
    
    Write-Host "🚀 Starting Mock Feed Server..." -ForegroundColor Yellow
    $feed = Start-Process python -ArgumentList "http_mock_feed.py" -PassThru
    $processes += @{Name="Mock Feed Server"; Process=$feed}
    Start-Sleep 3
    
    Write-Host "🚀 Starting Complete Trading System..." -ForegroundColor Yellow
    $trading = Start-Process python -ArgumentList "complete_trading_system.py" -PassThru
    $processes += @{Name="Trading System"; Process=$trading}
    Start-Sleep 5
    
    Write-Host "`n🔍 System Health Check:" -ForegroundColor Cyan
    Write-Host "-" * 30
    
    # Test endpoints
    try { 
        Invoke-RestMethod "http://localhost:5008" -TimeoutSec 2 | Out-Null
        Write-Host "✅ System Monitor is healthy" -ForegroundColor Green
    } catch { 
        Write-Host "❌ System Monitor not responding" -ForegroundColor Red 
    }
    
    try { 
        Invoke-RestMethod "http://localhost:5006" -TimeoutSec 2 | Out-Null
        Write-Host "✅ Analytics Dashboard is healthy" -ForegroundColor Green
    } catch { 
        Write-Host "❌ Analytics Dashboard not responding" -ForegroundColor Red 
    }
    
    try { 
        Invoke-RestMethod "http://localhost:8080" -TimeoutSec 2 | Out-Null
        Write-Host "✅ Mock Feed Server is healthy" -ForegroundColor Green
    } catch { 
        Write-Host "❌ Mock Feed Server not responding" -ForegroundColor Red 
    }
    
    Write-Host "`n🎉 All systems launched! Running $($processes.Count) processes" -ForegroundColor Green
    Write-Host "`n📊 Access your dashboards:" -ForegroundColor Cyan
    Write-Host "   • System Monitor:      http://localhost:5008" -ForegroundColor White
    Write-Host "   • Analytics Dashboard: http://localhost:5006" -ForegroundColor White
    Write-Host "   • Mock Feed Server:    http://localhost:8080" -ForegroundColor White
    
    Write-Host "`n⚡ Complete Trading System Architecture Active!" -ForegroundColor Magenta
    Write-Host "   Mock/Live Feed → Signal Generator → Order Brain → Risk Manager → Dashboard" -ForegroundColor White
    
    # Open dashboards in browser
    Write-Host "`n🌐 Opening dashboards in browser..." -ForegroundColor Yellow
    Start-Process "http://localhost:5008"  # System Monitor
    Start-Process "http://localhost:5006"  # Analytics Dashboard
    
    Write-Host "`n💡 Press Ctrl+C to shutdown all systems..." -ForegroundColor Yellow
    
    # Keep running until interrupted
    while ($true) {
        Start-Sleep 1
    }
    
} catch {
    Write-Host "`n🛑 Error occurred: $($_.Exception.Message)" -ForegroundColor Red
} finally {
    Write-Host "`n🛑 Shutting down all systems..." -ForegroundColor Yellow
    foreach ($proc in $processes) {
        if ($proc.Process -and !$proc.Process.HasExited) {
            Write-Host "   Stopping $($proc.Name)..." -ForegroundColor Yellow
            $proc.Process.Kill()
        }
    }
    Write-Host "✅ All systems stopped successfully!" -ForegroundColor Green
}