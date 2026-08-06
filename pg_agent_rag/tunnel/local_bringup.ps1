#Requires -Version 5.0
<#
.SYNOPSIS
  Local-side bring-up for PG-Agent RAG (Windows).
  1) SSH tunnel AutoDL :6006 -> localhost :18180
  2) Health + retrieve check
  3) Optional: list adb devices (does not start tasks)

.EXAMPLE
  .\local_bringup.ps1 -SshHost connect.nmb1.seetacloud.com -SshPort 46572
#>
param(
    [string]$SshHost = "connect.nmb1.seetacloud.com",
    [int]$SshPort = 46572,
    [int]$LocalPort = 18180,
    [int]$RemotePort = 6006,
    [switch]$SkipAdb
)

$ErrorActionPreference = "Stop"
$RagUrl = "http://127.0.0.1:$LocalPort"

# Kill stale listener on LocalPort if any (optional soft check)
$existing = Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Port $LocalPort already listening — assuming tunnel exists."
} else {
    Write-Host "Starting SSH tunnel in background..."
    $sshArgs = @(
        "-CNg",
        "-L", "${LocalPort}:127.0.0.1:${RemotePort}",
        "root@${SshHost}",
        "-p", "$SshPort",
        "-o", "ServerAliveInterval=30",
        "-o", "ExitOnForwardFailure=yes"
    )
    Start-Process -FilePath "ssh" -ArgumentList $sshArgs -WindowStyle Minimized
    Start-Sleep -Seconds 2
}

Write-Host "GET $RagUrl/health"
$health = Invoke-RestMethod -Uri "$RagUrl/health" -TimeoutSec 15
$health | ConvertTo-Json -Compress
if ($health.status -ne "ok") { throw "RAG health not ok" }

Write-Host "POST $RagUrl/retrieve"
$body = @{
    summary = "Android Settings page with Network and Wi-Fi"
    top_k = 4
    bfs_layers = 3
    max_guidelines = 5
} | ConvertTo-Json
$result = Invoke-RestMethod -Uri "$RagUrl/retrieve" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 60
Write-Host ("guidelines=" + $result.guidelines.Count)
if (-not $result.guidelines -or $result.guidelines.Count -eq 0) {
    throw "empty guidelines"
}

if (-not $SkipAdb) {
    $adb = Get-Command adb -ErrorAction SilentlyContinue
    if ($adb) {
        Write-Host "adb devices:"
        & adb devices
    } else {
        Write-Host "adb not in PATH (skip). Install platform-tools when ready."
    }
}

Write-Host ""
Write-Host "OK. Set for agent:"
Write-Host "  `$env:RAG_URL='$RagUrl'"
Write-Host "Then in android_world:"
Write-Host "  from android_world_hook import inject_guidelines"
