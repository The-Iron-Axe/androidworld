# PG-Agent RAG tunnel helpers for Windows (local PC <-> AutoDL)
#
# Usage (PowerShell):
#   .\start_tunnel.ps1
#   .\start_tunnel.ps1 -SshHost connect.nmb1.seetacloud.com -SshPort 44096
#
# Prefer tunnel/config.env for host/port (overrides stale shell env vars).
# Then set RAG_URL for the local agent:
#   $env:RAG_URL="http://127.0.0.1:18180"

param(
    [string]$SshHost,
    [int]$SshPort = 0,
    [int]$LocalPort = 0,
    [int]$RemotePort = 0,
    [string]$User = "root"
)

# Load tunnel/config.env first so it wins over stale $env:AUTODL_SSH_* in this shell
$configPath = Join-Path $PSScriptRoot "config.env"
if (Test-Path $configPath) {
    Get-Content $configPath | ForEach-Object {
        if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
        $k, $v = $_ -split '=', 2
        if ($k -and $v) {
            [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), "Process")
        }
    }
}

if (-not $SshHost) {
    $SshHost = $env:AUTODL_SSH_HOST
}
if ($SshPort -le 0 -and $env:AUTODL_SSH_PORT) {
    $SshPort = [int]$env:AUTODL_SSH_PORT
}
if ($LocalPort -le 0) {
    $LocalPort = if ($env:LOCAL_PORT) { [int]$env:LOCAL_PORT } else { 18180 }
}
if ($RemotePort -le 0) {
    $RemotePort = if ($env:REMOTE_PORT) { [int]$env:REMOTE_PORT } else { 6006 }
}

if (-not $SshHost -or $SshPort -le 0) {
    Write-Error @"
Missing AutoDL SSH info.
Provide -SshHost / -SshPort (from AutoDL console SSH command), e.g.:
  .\start_tunnel.ps1 -SshHost connect.nmb1.seetacloud.com -SshPort 44096
Or copy config.example.env to config.env and fill AUTODL_SSH_HOST / AUTODL_SSH_PORT.
"@
    exit 1
}

Write-Host "Tunnel: localhost:$LocalPort -> ${User}@${SshHost}:$SshPort -> 127.0.0.1:$RemotePort"
Write-Host "After connect, test: curl http://127.0.0.1:$LocalPort/health"
Write-Host "Keep this window open while the agent runs."

ssh -CNg -L "${LocalPort}:127.0.0.1:${RemotePort}" "${User}@${SshHost}" -p $SshPort
