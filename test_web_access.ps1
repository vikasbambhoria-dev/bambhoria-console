# Test bambhoriaquantum.in access
Write-Host "Testing Bambhoria Quantum Web Application..."
Write-Host "================================================"

# Test local access
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:5000" -UseBasicParsing
    if ($response.Content -match "BAMBHORIA QUANTUM") {
        Write-Host "✅ SUCCESS: 'BAMBHORIA QUANTUM' found on homepage!" -ForegroundColor Green
        Write-Host "🌐 Domain title is showing properly" -ForegroundColor Green
    } else {
        Write-Host "❌ ERROR: 'BAMBHORIA QUANTUM' NOT found on homepage!" -ForegroundColor Red
        Write-Host "🔍 Page content preview:" -ForegroundColor Yellow
        Write-Host $response.Content.Substring(0, [Math]::Min(500, $response.Content.Length))
    }
} catch {
    Write-Host "❌ ERROR: Cannot access web application" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "📋 MANUAL TEST INSTRUCTIONS:"
Write-Host "1. Open browser and go to: http://127.0.0.1:5000"
Write-Host "2. You should see: 'BAMBHORIA QUANTUM' title"
Write-Host "3. Domain should show: 'bambhoriaquantum.in'"
Write-Host ""
Write-Host "🔧 For bambhoriaquantum.in domain:"
Write-Host "1. Run Command Prompt as Administrator"
Write-Host "2. Run: notepad C:\Windows\System32\drivers\etc\hosts"
Write-Host "3. Add line: 127.0.0.1 bambhoriaquantum.in"
Write-Host "4. Save and visit: http://bambhoriaquantum.in"