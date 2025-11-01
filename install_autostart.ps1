# Requires -RunAs Administrator
# Purpose: Install Windows Scheduled Task to auto-start Bambhoria Quantum server at user logon,
# ensure hosts entry and portproxy (80->5000), and optional firewall rule.

param(
    [string]$ProjectDir = (Split-Path -Parent $MyInvocation.MyCommand.Path),
    [string]$TaskName = "BambhoriaQuantumServer",
    [string]$DomainHost = "bambhoriaquantum.in",
    [string]$Listen = "127.0.0.1:5000"
)

function Ensure-Admin {
    $currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentIdentity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Host "Elevating to administrator..."
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = (Get-Process -Id $PID).Path
        $psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$($MyInvocation.MyCommand.Path)`""
        $psi.Verb = "runas"
        [Diagnostics.Process]::Start($psi) | Out-Null
        exit
    }
}

function Ensure-HostsEntry($host) {
    $hosts = "$env:WinDir\System32\drivers\etc\hosts"
    $content = Get-Content $hosts -ErrorAction SilentlyContinue
    if ($content -notmatch "\s$host(\s|$)") {
        Write-Host "Adding hosts entry for $host -> 127.0.0.1"
        Add-Content -Path $hosts -Value "127.0.0.1 $host"
    } else {
        Write-Host "Hosts entry for $host already present"
    }
}

function Ensure-PortProxy($listen, $toAddress="127.0.0.1", $toPort=5000) {
    $parts = $listen.Split(":")
    $laddr = $parts[0]
    $lport = [int]$parts[1]
    $rules = netsh interface portproxy show v4tov4 | Out-String
    if ($rules -notmatch "\s$laddr\s+$lport\s+$toAddress\s+$toPort\s*") {
        Write-Host "Creating portproxy $laddr:$lport -> $toAddress:$toPort"
        netsh interface portproxy add v4tov4 listenport=$lport listenaddress=$laddr connectport=$toPort connectaddress=$toAddress | Out-Null
    } else {
        Write-Host "Portproxy already configured"
    }
}

function Ensure-Firewall($name, $port) {
    $rule = Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue
    if (-not $rule) {
        New-NetFirewallRule -DisplayName $name -Direction Inbound -Action Allow -Protocol TCP -LocalPort $port | Out-Null
        Write-Host "Firewall rule $name added for port $port"
    } else {
        Write-Host "Firewall rule $name already exists"
    }
}

function Install-Task($taskName, $projectDir) {
    $python = Join-Path $projectDir ".venv\\Scripts\\python.exe"
    if (-not (Test-Path $python)) {
        throw ".venv not found. Create venv and install dependencies before installing task."
    }
    $action = New-ScheduledTaskAction -Execute $python -Argument "-m waitress --listen=127.0.0.1:5000 wsgi:application" -WorkingDirectory $projectDir
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
    $task = New-ScheduledTask -Action $action -Trigger $trigger -Principal $principal -Settings $settings
    Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null
    Write-Host "Scheduled task $taskName installed."
}

# Main
Ensure-Admin
Ensure-HostsEntry -host $DomainHost
Ensure-PortProxy -listen "127.0.0.1:80" -toAddress "127.0.0.1" -toPort 5000
# Optional firewall for local access
Ensure-Firewall -name "Bambhoria Quantum 5000" -port 5000
Install-Task -taskName $TaskName -projectDir $ProjectDir
Write-Host "Done. The server will auto-start at logon. You can manually start it now from Task Scheduler if needed."