param(
    [switch]$ProcessBacklog
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
$agent = Join-Path $PSScriptRoot "windows_visio_agent.py"
$pidFile = Join-Path $repo "logs\windows_visio_agent.pid"
$outLog = Join-Path $repo "logs\windows_visio_agent.out.log"
$errLog = Join-Path $repo "logs\windows_visio_agent.err.log"

if (-not (Test-Path $python)) {
    throw "Python virtual environment not found: $python"
}

if (Test-Path $pidFile) {
    $existingPid = (Get-Content $pidFile -Raw).Trim()
    if ($existingPid -and (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) {
        Write-Host "Windows Visio Agent is already running with PID $existingPid."
        exit 0
    }
    Remove-Item $pidFile -Force
}

if (-not $ProcessBacklog) {
    Write-Warning "The agent processes all files currently in SharePoint Pending_Draw."
    Write-Warning "Pass -ProcessBacklog to confirm that behavior."
    exit 2
}

$process = Start-Process `
    -FilePath $python `
    -ArgumentList $agent `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError $errLog `
    -WindowStyle Hidden `
    -PassThru

Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ascii
Write-Host "Windows Visio Agent started with PID $($process.Id)."
Write-Host "Output log: $outLog"
Write-Host "Error log:  $errLog"
