# Verify SSH local-forward from Windows after start_tunnel.* is running.
# Run on local PC:
#   powershell -File verify_tunnel.ps1

param(
    [string]$RagUrl = $(if ($env:RAG_URL) { $env:RAG_URL } else { "http://127.0.0.1:18180" })
)

Write-Host "GET $RagUrl/health"
try {
    $health = Invoke-RestMethod -Uri "$RagUrl/health" -Method Get -TimeoutSec 10
    $health | ConvertTo-Json -Depth 5
} catch {
    Write-Error "Tunnel/health failed: $_"
    exit 1
}

Write-Host "POST $RagUrl/retrieve"
$body = @{
    summary = "Android Settings page with Wi-Fi and Network options"
    top_k = 4
    bfs_layers = 3
    max_guidelines = 10
} | ConvertTo-Json

try {
    $result = Invoke-RestMethod -Uri "$RagUrl/retrieve" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 60
    $result | ConvertTo-Json -Depth 8
    if (-not $result.guidelines -or $result.guidelines.Count -eq 0) {
        Write-Error "retrieve returned empty guidelines"
        exit 1
    }
    Write-Host "Tunnel verify OK"
} catch {
    Write-Error "retrieve failed: $_"
    exit 1
}
