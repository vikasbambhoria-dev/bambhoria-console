param(
    [string]$Name = "New Bambhoria Strategy",
    [string]$Model = "lightgbm_v51"
)

Write-Host "🪶 New Bambhoria Strategy Scaffolder"
Write-Host "-----------------------------------"
if (-not $PSBoundParameters.ContainsKey('Name')) {
    $Name = Read-Host "Enter Strategy Name (default: $Name)"
    if (-not $Name) { $Name = "New Bambhoria Strategy" }
}

# Ensure we run from repo root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
Set-Location $repoRoot

# Create scaffold via composer
python ai_engine/omniverse_strategy_composer.py --non-interactive --scaffold --name "$Name" --model "$Model"

# Show the created file
$fname = ($Name -replace ' ', '_') + ".json"
$path = Join-Path "$repoRoot/strategies" $fname
if (Test-Path $path) {
    Write-Host "✅ Created: $path"
    Get-Content $path | Select-Object -First 20 | Write-Output
} else {
    Write-Warning "Scaffold did not produce expected file: $path"
}
