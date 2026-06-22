param(
    [string]$Server = "root@46.62.131.88",
    [string]$RemoteDir = "/opt/Plangenration",
    [switch]$AllowDirty
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$branch = (git branch --show-current).Trim()
$sha = (git rev-parse HEAD).Trim()
$dirty = git status --porcelain

if ($branch -ne "main") {
    throw "Deployment must run from main; current branch is '$branch'."
}
if ($dirty -and -not $AllowDirty) {
    throw "Working tree is dirty. Commit and push changes before deployment."
}

git fetch origin main
$originSha = (git rev-parse origin/main).Trim()
if ($sha -ne $originSha) {
    throw "Local main ($sha) does not match origin/main ($originSha). Push first."
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backup = "$RemoteDir/backups/$stamp"

Write-Host "Deploying Git commit $sha to $Server..."
ssh $Server "mkdir -p '$backup' && cp '$RemoteDir/process_plan.py' '$backup/' && cp '$RemoteDir/generate_plan.py' '$backup/' && cp '$RemoteDir/Dockerfile' '$backup/' && cp '$RemoteDir/template.vsdx' '$backup/' && if [ -f '$RemoteDir/layout_extractor.py' ]; then cp '$RemoteDir/layout_extractor.py' '$backup/'; fi && if [ -f '$RemoteDir/vision_client.py' ]; then cp '$RemoteDir/vision_client.py' '$backup/'; fi"

scp automation/container/process_plan.py "${Server}:$RemoteDir/process_plan.py"
scp automation/container/generate_plan.py "${Server}:$RemoteDir/generate_plan.py"
scp automation/container/Dockerfile "${Server}:$RemoteDir/Dockerfile"
scp automation/container/template.vsdx "${Server}:$RemoteDir/template.vsdx"
scp utils/layout_extractor.py "${Server}:$RemoteDir/layout_extractor.py"
scp utils/vision_client.py "${Server}:$RemoteDir/vision_client.py"

ssh $Server "cd '$RemoteDir' && docker build --build-arg PLAN_GIT_SHA='$sha' -t 'plangenration:$sha' -t plangenration:latest ."
ssh $Server "docker stop plangenration && docker rm plangenration && docker run -d --name plangenration --restart unless-stopped --env-file /opt/acorn/.env --network acorn_network -v /opt/Plangenration/reports:/app/src/output/reports -v /opt/acorn/config:/app/config:ro plangenration:latest"
ssh $Server "docker exec plangenration python src/process_plan.py --help >/dev/null && docker image inspect plangenration:latest --format '{{json .Config.Labels}}'"

Write-Host "Deployment complete. Running plangenration revision: $sha"
