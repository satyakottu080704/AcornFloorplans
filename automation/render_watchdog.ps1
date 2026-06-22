# render_watchdog.ps1 — keep the Acorn render service alive.
# Runs every minute in the logged-on session (Visio COM needs an interactive
# desktop). If nothing is listening on the render port, (re)start the service.
#
# Install (run once, in admin PowerShell, while logged in as the auto-logon user):
#   $a = New-ScheduledTaskAction -Execute 'powershell.exe' `
#        -Argument '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File C:\AcornPlanGeneration\automation\render_watchdog.ps1'
#   $t = New-ScheduledTaskTrigger -AtLogOn
#   $t.Repetition = (New-ScheduledTaskTrigger -Once -At (Get-Date) `
#        -RepetitionInterval (New-TimeSpan -Minutes 1) `
#        -RepetitionDuration (New-TimeSpan -Days 3650)).Repetition
#   $p = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
#   Register-ScheduledTask -TaskName 'Acorn Render Watchdog' -Action $a -Trigger $t -Principal $p -Force

$ErrorActionPreference = 'SilentlyContinue'
$port    = 8765
$root    = 'C:\AcornPlanGeneration'
$python  = Join-Path $root '.venv\Scripts\python.exe'
$script  = 'automation\render_service.py'

$listening = Get-NetTCPConnection -LocalPort $port -State Listen
if (-not $listening) {
    if (Test-Path $python) {
        Start-Process -FilePath $python -ArgumentList $script -WorkingDirectory $root -WindowStyle Hidden
    } else {
        # Fall back to whatever 'python' is on PATH if the venv is missing.
        Start-Process -FilePath 'python' -ArgumentList $script -WorkingDirectory $root -WindowStyle Hidden
    }
}
