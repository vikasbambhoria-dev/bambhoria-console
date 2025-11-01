param()

# Pre-commit hook: validate staged strategy JSON files
# Usage: Install via scripts/install_precommit_hook.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[pre-commit] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[pre-commit] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[pre-commit] $msg" -ForegroundColor Red }

# Determine python executable
$python = Join-Path -Path $PSScriptRoot -ChildPath '..\..\.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }

# Get staged files
$staged = git diff --cached --name-only --diff-filter=ACM
if ($LASTEXITCODE -ne 0) {
  Write-Err "Failed to list staged files"
  exit 2
}

$files = @()
foreach ($f in $staged) {
  if ($f -like 'strategies/*.json' -or $f -like 'strategies\\*.json') {
    $files += $f
  }
}

if ($files.Count -eq 0) {
  Write-Info "No strategy JSON files staged; skipping"
  exit 0
}

$status = 0
foreach ($f in $files) {
  Write-Info "Validating $f"
  & $python ai_engine/omniverse_strategy_composer.py --validate-file "$f" --meta-validate --strict
  if ($LASTEXITCODE -ne 0) {
    Write-Err "Validation failed for $f"
    $status = 1
  }
}

if ($status -ne 0) {
  Write-Err "Commit blocked: strategy validation failed"
} else {
  Write-Info "All staged strategy files valid"
}
exit $status
