param(
    [switch]$NoVenv,
    [switch]$NoOpen,
    [switch]$NoWatchdog
)

# Runs the Complete System Launcher with proper environment
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Resolve repo root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Get-Item (Join-Path $scriptDir '..')).FullName
Set-Location $repoRoot

Write-Host "Repo: $repoRoot" -ForegroundColor Cyan

# Activate venv unless disabled
$venvActivate = Join-Path $repoRoot '.venv/Scripts/Activate.ps1'
if (-not $NoVenv -and (Test-Path $venvActivate)) {
    Write-Host "Activating venv..." -ForegroundColor DarkCyan
    . $venvActivate
} else {
    Write-Warning "Virtual environment not activated (use -NoVenv to skip)."
}

# Ensure UTF-8 console for emojis/logs
$env:PYTHONIOENCODING = 'utf-8'

# Create logs folder if missing
$logs = Join-Path $repoRoot 'logs'
if (-not (Test-Path $logs)) { New-Item -ItemType Directory -Path $logs | Out-Null }

# Build optional flags
$flags = @()
if ($NoOpen) { $flags += '--no-open' }
if ($NoWatchdog) { $flags += '--no-watchdog' }

# Launch the complete system (blocks until Ctrl+C)
Write-Host "Launching Bambhoria Complete System Launcher (v2)..." -ForegroundColor Green
python complete_system_launcher_v2.py @flags
