#!/usr/bin/env bash
# Print the exact ssh -L command for the current AutoDL instance.
# On Windows, prefer tunnel/start_tunnel.ps1 with the same host/port.

HOST="${AUTODL_SSH_HOST:-connect.nmb1.seetacloud.com}"
PORT="${AUTODL_SSH_PORT:-46572}"
LOCAL="${LOCAL_PORT:-18180}"
REMOTE="${REMOTE_PORT:-6006}"

cat <<EOF
# Run on your LOCAL Windows / Mac / Linux PC (not inside AutoDL):

ssh -CNg -L ${LOCAL}:127.0.0.1:${REMOTE} root@${HOST} -p ${PORT}

# Then:
export RAG_URL=http://127.0.0.1:${LOCAL}
curl \$RAG_URL/health

# PowerShell:
#   .\\start_tunnel.ps1 -SshHost ${HOST} -SshPort ${PORT}
#   .\\verify_tunnel.ps1
EOF
