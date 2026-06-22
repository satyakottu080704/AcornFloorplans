$repo = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $repo "logs\windows_visio_agent.pid"
$outLog = Join-Path $repo "logs\windows_visio_agent.out.log"
$errLog = Join-Path $repo "logs\windows_visio_agent.err.log"
$productionLog = Join-Path $repo "logs\windows_visio_agent.production.log"
$taskName = "Acorn Windows Visio Agent"

$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task) {
    Write-Host "Production Scheduled Task '$taskName' state: $($task.State)"
    if (Test-Path $productionLog) {
        Write-Host "`nRecent production output:"
        Get-Content $productionLog -Tail 20
    }
    if ($task.State -eq "Running") {
        exit 0
    }
    exit 1
}

if (-not (Test-Path $pidFile)) {
    Write-Host "Windows Visio Agent is not running."
    exit 1
}

$agentPid = (Get-Content $pidFile -Raw).Trim()
$process = Get-Process -Id $agentPid -ErrorAction SilentlyContinue
if (-not $process) {
    Write-Host "Windows Visio Agent is not running. Stale PID file: $agentPid"
    exit 1
}

Write-Host "Windows Visio Agent is running with PID $agentPid."
if (Test-Path $outLog) {
    Write-Host "`nRecent output:"
    Get-Content $outLog -Tail 20
}
if (Test-Path $errLog) {
    Write-Host "`nRecent errors:"
    Get-Content $errLog -Tail 10
}
