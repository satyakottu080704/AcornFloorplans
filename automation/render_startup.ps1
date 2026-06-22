# render_startup.ps1 — run automatically at logon (auto-logon session).
# Deploys the latest code, then starts the render service. The watchdog
# (render_watchdog.ps1) separately restarts the service if it ever dies.
#
# One-time install (point the existing "Acorn Render Service" task at this):
#   $a = New-ScheduledTaskAction -Execute 'powershell.exe' `
#        -Argument '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File C:\AcornPlanGeneration\automation\render_startup.ps1'
#   Set-ScheduledTask -TaskName 'Acorn Render Service' -Action $a

$ErrorActionPreference = 'Continue'
$root = 'C:\AcornPlanGeneration'
$log  = Join-Path $root 'startup.log'
Set-Location $root

"[startup $(Get-Date -Format s)] pulling latest code..." | Out-File -Append $log
# Deploy latest code (autostash so local junk never blocks the pull).
& git pull --rebase --autostash *>> $log

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }

"[startup $(Get-Date -Format s)] starting render service..." | Out-File -Append $log
# Run the service in THIS session (blocks here; Visio COM needs the desktop).
& $python 'automation\render_service.py' *>> $log
