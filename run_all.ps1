
Write-Host "🚀 Starting God-Eye Full System (Mock + Engine + Dashboard) - DO BEST" -ForegroundColor Cyan
$env:GODEYE_MODEL_PATH = "models\godeye_model.pkl"
$env:GODEYE_WS_URL = "ws://127.0.0.1:8765/ws"
$env:GODEYE_DATA_WS = "ws://127.0.0.1:8765/ws"
# ensure venv exists
if (-not (Test-Path ".\.venv")) {
    python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
# install minimal deps if missing
$req = "requirements.txt"
if (Test-Path $req) {
    try { pip install -r $req } catch { Write-Warning "pip install failed - ensure dependencies installed manually." }
}
Start-Process powershell -ArgumentList "-NoExit","-Command","python godeye_mock_server_advanced.py --http-port 8000 --ws-port 8765 --interval 1.0"
Start-Sleep -Seconds 3
Start-Process powershell -ArgumentList "-NoExit","-Command","python main.py --mode auto"
Start-Sleep -Seconds 3
Start-Process powershell -ArgumentList "-NoExit","-Command","python dashboard/dashboard_server.py --http-port 5000"
Start-Sleep -Seconds 2
# open default browser
$browser = "msedge.exe"
if (-not (Get-Command $browser -ErrorAction SilentlyContinue)) { $browser = "chrome.exe" }
Start-Process $browser "http://localhost:5000"
Write-Host "✅ All modules launched! (close windows to stop)" -ForegroundColor Green
