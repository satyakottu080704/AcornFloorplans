$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
$agent = Join-Path $PSScriptRoot "windows_visio_agent.py"
$logDir = Join-Path $repo "logs"
$logFile = Join-Path $logDir "windows_visio_agent.production.log"

New-Item -ItemType Directory -Path $logDir -Force | Out-Null

"$(Get-Date -Format o) Starting Windows Visio Agent" | Add-Content -LiteralPath $logFile
& $python -u $agent *>> $logFile
$exitCode = $LASTEXITCODE
"$(Get-Date -Format o) Windows Visio Agent exited with code $exitCode" | Add-Content -LiteralPath $logFile
exit $exitCode
