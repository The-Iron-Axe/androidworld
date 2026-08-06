# Source on local PC after SSH tunnel is up (bash / Git Bash / WSL):
#   source client/set_rag_env.sh
# Then run your android_world / PG-Agent process in the same shell.

export RAG_URL="${RAG_URL:-http://127.0.0.1:18180}"
echo "RAG_URL=$RAG_URL"
# Quick check (optional):
# curl -sS "$RAG_URL/health"
