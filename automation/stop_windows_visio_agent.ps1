$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $repo "logs\windows_visio_agent.pid"

if (-not (Test-Path $pidFile)) {
    Write-Host "Windows Visio Agent is not running."
    exit 0
}

$agentPid = (Get-Content $pidFile -Raw).Trim()
$process = Get-Process -Id $agentPid -ErrorAction SilentlyContinue
if ($process) {
    Stop-Process -Id $agentPid
    Write-Host "Stopped Windows Visio Agent PID $agentPid."
}
Remove-Item $pidFile -Force
