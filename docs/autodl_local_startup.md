# AutoDL + 本机启动指南（U3 / PG-Agent RAG）

两头分工：

| 位置 | 职责 |
|------|------|
| **AutoDL** | 跑 Page-Graph RAG 服务（BGE-M3 + FAISS），监听 `6006` |
| **本机 Windows** | SSH 隧道、AVD、android_world / `test_u1_u2.py` |

```
本机 agent  --HTTP-->  127.0.0.1:18180  --ssh -L-->  AutoDL:6006  (RAG)
本机 agent  --ADB--->  本地模拟器
```

---

## A. AutoDL（云端）

在 AutoDL **JupyterLab / 终端**里操作。

### A1. 开机后启动 RAG（每次实例开机都要做）

```bash
export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT=https://hf-mirror.com

cd /root/pg_agent_rag

# 仅首次，或换了 page_graph.json 之后：
# bash scripts/build_index.sh

bash scripts/start_server.sh
```

看到类似输出即成功：

```text
Starting RAG server on 0.0.0.0:6006
INFO:     Uvicorn running on http://0.0.0.0:6006
```

**这个终端保持开着，不要 Ctrl+C。**

### A2.（可选）同机自检

另开一个 AutoDL 终端：

```bash
curl -sS http://127.0.0.1:6006/health
# 或
bash /root/pg_agent_rag/scripts/smoke_test.sh
```

### A3. 复制本机要用的 SSH 信息

AutoDL 控制台 → 该实例 → **SSH 指令**，例如：

```text
ssh -p 44096 root@connect.nmb1.seetacloud.com
```

记下：

- 主机：`connect.nmb1.seetacloud.com`
- 端口：`44096`（**每次开机可能变**）

---

## B. 本机 Windows

项目根目录：`C:\Users\WRQ\Desktop\androidworld`  
模拟器、ADB、评测脚本都在本机跑。

### B0. 写好隧道配置（SSH 端口变了就改）

编辑 `pg_agent_rag\tunnel\config.env`：

```env
AUTODL_SSH_HOST=connect.nmb1.seetacloud.com
AUTODL_SSH_PORT=44096
LOCAL_PORT=18180
REMOTE_PORT=6006
RAG_URL=http://127.0.0.1:18180
```

（无此文件时：复制 `config.example.env` → `config.env` 再改。）

### B1. 终端① — SSH 隧道（必须一直开着）

```powershell
cd C:\Users\WRQ\Desktop\androidworld\pg_agent_rag\tunnel
.\start_tunnel.ps1
```

成功时大致显示：

```text
Tunnel: localhost:18180 -> root@connect.nmb1.seetacloud.com:44096 -> 127.0.0.1:6006
```

窗口会挂住（像卡住一样）是正常的。若 `Connection refused`，回 A3 更新端口。

也可显式指定：

```powershell
.\start_tunnel.ps1 -SshHost connect.nmb1.seetacloud.com -SshPort 44096
```

### B2. 终端② — 验证隧道

```powershell
cd C:\Users\WRQ\Desktop\androidworld\pg_agent_rag\tunnel
.\verify_tunnel.ps1
```

末尾出现 `Tunnel verify OK` 即可。

设置环境变量（同一终端后续跑评测也行）：

```powershell
. ..\client\set_rag_env.ps1
# 或： $env:RAG_URL = "http://127.0.0.1:18180"
```

### B3. 终端③ — 模拟器（若尚未开）

用 Android Studio / 命令行启动 AVD（例如 Pixel_6），确认：

```powershell
& "D:\Data\Android\platform-tools\adb.exe" devices
# 应看到 device 而不是 offline
```

### B4. 终端④ — 跑 U3 评测（+U1+U2+U3）

```powershell
cd C:\Users\WRQ\Desktop\androidworld
$env:RAG_URL = "http://127.0.0.1:18180"

python scripts/test_u1_u2.py --u3 --tasks=SimpleCalendarAddOneEvent
```

- `--u3` = 单轮 **+U1+U2+U3**（phase 名 `u1u2u3`）
- 换任务：改 `--tasks=MarkorMergeNotes` 等
- 多任务：`--tasks=T1,T2`

结果写在：`scripts/results/<run_id>_u1u2u3.json`

---

## 启动顺序（最短清单）

1. **AutoDL**：`bash scripts/start_server.sh`（保持）
2. **本机①**：`.\start_tunnel.ps1`（保持）
3. **本机②**：`.\verify_tunnel.ps1` → OK
4. **本机**：开 AVD
5. **本机**：`python scripts/test_u1_u2.py --u3 --tasks=...`

---

## 常见问题

| 现象 | 处理 |
|------|------|
| 本机 `18180` 连接被拒绝 | 先跑 B1 隧道；确认 AutoDL 服务在跑 |
| `ssh ... Connection refused` | 控制台 SSH 端口已变，更新 `config.env` |
| `U3 RAG retrieve failed` | 隧道断了或 AutoDL 服务停了；重开 A1+B1 |
| 隧道 OK 但评测无 U3 块 | 确认跑评测的终端设置了 `RAG_URL` |
| AutoDL 换实例 / 关机 | 重新 A1，并更新本机 SSH 端口 |

---

## API 对照

| 本机访问 | 实际落到 |
|----------|----------|
| `GET http://127.0.0.1:18180/health` | AutoDL `:6006/health` |
| `POST http://127.0.0.1:18180/retrieve` | AutoDL `:6006/retrieve` |
