# PG-Agent RAG (AutoDL <-> Local AVD)

本目录是 PG-Agent 复现的**本机侧**工具包：通过 SSH 隧道调用 AutoDL 上的 Page-Graph RAG 服务，再把检索到的 guidelines 注入本地 `android_world` agent 循环。

相关文档：

- **启动指南（两头指令）**：[`docs/autodl_local_startup.md`](../docs/autodl_local_startup.md)
- 通信方案：[`docs/autodl_pg_agent_rag_plan.md`](../docs/autodl_pg_agent_rag_plan.md)
- 论文摘录：[`docs/paper/PG-Agent_preview.txt`](../docs/paper/PG-Agent_preview.txt)、[`docs/paper/PG-Agent_rest.txt`](../docs/paper/PG-Agent_rest.txt)
- 项目根目录也有一份同内容的 [`client/`](../client/)（便于 `PYTHONPATH` 引用）

## Layout

```
pg_agent_rag/
  client/          # RagClient + android_world hook（本机）
  tunnel/          # Windows SSH -L 助手
  README.md

# AutoDL 上另有（不在本仓库）：
#   server/        # FastAPI: /health /retrieve /embed
#   scripts/       # build_index / start_server / smoke_test
```

## AutoDL（云端 RAG 服务）

在 AutoDL 实例上（路径按你部署的为准，常见为 `/root/pg_agent_rag`）：

```bash
export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT=https://hf-mirror.com
cd /root/pg_agent_rag
bash scripts/build_index.sh      # 首次：demo page graph + FAISS
bash scripts/start_server.sh     # 监听 0.0.0.0:6006
# 另一个终端：
bash scripts/smoke_test.sh
```

用真实 page graph 替换 `/root/autodl-tmp/pg_agent_rag/data/page_graph.json` 后，重新跑 `build_index.sh`。

## 本机 Windows（AVD + android_world）

**隧道本身不会驱动模拟器**；只有你的 agent 循环才会调用 ADB。若机器上已有其他 AVD 任务在跑，先不要启动 agent。

1. 保持 AutoDL RAG 服务运行（`bash scripts/start_server.sh`）。
2. 用 AutoDL 控制台里的 SSH 主机/端口开隧道：

```powershell
cd pg_agent_rag\tunnel
# 可先复制 config.example.env -> config.env 填入 AUTODL_SSH_HOST / AUTODL_SSH_PORT
.\start_tunnel.ps1
# 或显式传参：
# .\start_tunnel.ps1 -SshHost connect.nmb1.seetacloud.com -SshPort 46572
```

3. 另开终端验证并设置 `RAG_URL`：

```powershell
cd pg_agent_rag\tunnel
.\verify_tunnel.ps1
. ..\client\set_rag_env.ps1   # RAG_URL=http://127.0.0.1:18180
```

一键拉起（隧道 + health/retrieve + 可选 adb devices）：

```powershell
cd pg_agent_rag\tunnel
.\local_bringup.ps1 -SshHost <host> -SshPort <port>
```

4. 接到桌面 AndroidWorld 的 **U3**（与 U1/U2 相同，走 `scripts/test_u1_u2.py`）：隧道保持开启后，

```powershell
$env:RAG_URL = "http://127.0.0.1:18180"
python scripts/test_u1_u2.py --u3 --tasks=SystemWifiTurnOn
# phase = u1u2u3  →  +U1+U2+U3
```

（`run.py --agent_name=m3a_qwen3_vl_32b_mem --u3` 也可，但日常评测用 test 脚本。）

每步会用当前 UI 文本构造 screen summary，调用 AutoDL `/retrieve`，把 page-graph guidelines 注入 action prompt（`## Environment Knowledge (U3)`）。

手动 hook（调试用）见 `client/integration_snippet.py`。本机联调：

```powershell
python client\local_avd_e2e.py
```

## API

- `GET /health`
- `POST /retrieve` `{"summary","top_k","bfs_layers","max_guidelines"}`
- `POST /embed` `{"texts":[...]}`

默认本机入口：`RAG_URL=http://127.0.0.1:18180`（经 `ssh -L 18180:127.0.0.1:6006` 转到 AutoDL）。
