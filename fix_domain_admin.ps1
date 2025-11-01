# Admin PowerShell script to fix bambhoriaquantum.in domain
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "    BAMBHORIA QUANTUM DOMAIN FIX" -ForegroundColor Yellow
Write-Host "    Adding bambhoriaquantum.in to hosts file" -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

try {
    # Check if running as administrator
    $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    $isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    
    if (-not $isAdmin) {
        Write-Host "ERROR: Not running as Administrator!" -ForegroundColor Red
        Write-Host "Please right-click and 'Run as Administrator'" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    $hostsFile = "C:\Windows\System32\drivers\etc\hosts"
    $domainEntry = "127.0.0.1 bambhoriaquantum.in"
    
    # Check if entry already exists
    $hostsContent = Get-Content $hostsFile -ErrorAction Stop
    if ($hostsContent -contains $domainEntry) {
        Write-Host "Domain mapping already exists!" -ForegroundColor Green
    } else {
        # Add the domain entry
        Add-Content -Path $hostsFile -Value "" -ErrorAction Stop
        Add-Content -Path $hostsFile -Value "# Bambhoria Quantum Local Domain" -ErrorAction Stop
        Add-Content -Path $hostsFile -Value $domainEntry -ErrorAction Stop
        
        Write-Host "SUCCESS: Domain mapping added!" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "You can now access:" -ForegroundColor Cyan
    Write-Host "http://bambhoriaquantum.in" -ForegroundColor Yellow
    Write-Host "https://bambhoriaquantum.in" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Web application is running on:" -ForegroundColor Cyan
    Write-Host "http://127.0.0.1:5000" -ForegroundColor Yellow
    Write-Host ""
    
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "Press Enter to close..." -ForegroundColor Gray
Read-Host