# Run on local Windows PowerShell after tunnel is up:
#   . .\client\set_rag_env.ps1
# Then start your android_world / PG-Agent in the same session.

$env:RAG_URL = if ($env:RAG_URL) { $env:RAG_URL } else { "http://127.0.0.1:18180" }
Write-Host "RAG_URL=$env:RAG_URL"
# Optional: Invoke-RestMethod "$env:RAG_URL/health"
