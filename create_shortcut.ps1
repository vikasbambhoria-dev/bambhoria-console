
param([string]$Name = "Run God-Eye Suite")
$ws = (Get-Location).Path
$Target = "$ws\run_all.ps1"
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\$Name.lnk")
$Shortcut.TargetPath = "powershell.exe"
$Shortcut.Arguments = "-NoExit -ExecutionPolicy Bypass -File `"$Target`""
$Shortcut.WorkingDirectory = $ws
$Shortcut.Save()
Write-Host "Shortcut created on Desktop: $env:USERPROFILE\Desktop\$Name.lnk"
