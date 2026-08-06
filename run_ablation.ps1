# ============================================================
# run_ablation.ps1 — 一键运行 U1-U4 层次累加消融（两种子，共享 store）
#
# 流程（共享 store 方案）：
#   Stage A  积累: 全开记忆, 在 record_tasks 上跑 record_rounds 轮
#                  (DMS 多轮成熟; 每轮复用上轮 store)
#   Stage C  消融: u12 -> u123 -> u1234 在 eval_tasks 上评估,
#                  读同一批成熟 store, 差异归因于启用层
#   两个 seed: 每个 seed 独立一套 store (u2_store/seed30/ 等),
#              末尾输出各配置跨 seed 的均值±std
#
# 前置: AutoDL RAG 服务已启动 + 本机 SSH 隧道 (127.0.0.1:18180) 已就绪
# 用法: powershell -ExecutionPolicy Bypass -File run_ablation.ps1
# ============================================================

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# ────────────────────────── 配置区（按需修改）──────────────────────────
$RecordTasks = "MarkorCreateNoteAndSms,MarkorMergeNotes,ExpenseAddMultipleFromMarkor"
$EvalTasks   = "MarkorTranscribeVideo,RecipeAddMultipleRecipesFromMarkor2,ExpenseDeleteMultiple2"
$Seeds       = "30,31"
$RecordRounds = 3
$RagUrl      = "http://127.0.0.1:18180"
# ──────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "=== 一键消融: 共享 store / 两种子 / 层次累加 ==="
Write-Host "  积累任务 : $RecordTasks"
Write-Host "  评估任务 : $EvalTasks"
Write-Host "  种子     : $Seeds   (每 seed 独立 store)"
Write-Host "  积累轮数 : $RecordRounds"
Write-Host ""

# ── 前置检查: RAG 隧道 (U3 是消融独立变量, 隧道断则 U3 结果不可信) ──
Write-Host "==> 检查 RAG 隧道: $RagUrl/health"
try {
  $health = Invoke-RestMethod -Uri "$RagUrl/health" -TimeoutSec 10
  Write-Host "    RAG OK: $($health.status)"
} catch {
  Write-Error @"
RAG 隧道不可用: $($_.Exception.Message)

修复步骤:
  1. AutoDL 上: bash /root/pg_agent_rag/scripts/start_server.sh  (保持终端开着)
  2. 本机:      .\pg_agent_rag\tunnel\start_tunnel.ps1           (保持窗口挂着)
  3. 验证:      .\pg_agent_rag\tunnel\verify_tunnel.ps1  -> Tunnel verify OK
然后重跑本脚本。
"@
  exit 1
}

# ── 运行消融 ──
Write-Host ""
Write-Host "==> 启动消融 (scripts/ablation_hierarchical.py) ..."
& python scripts/ablation_hierarchical.py `
    --record_tasks=$RecordTasks `
    --tasks=$EvalTasks `
    --seeds=$Seeds `
    --record_rounds=$RecordRounds `
    --rag_url=$RagUrl `
    --rag_on

if ($LASTEXITCODE -ne 0) {
  Write-Error "消融运行失败 (exit=$LASTEXITCODE)。见上方报错。"
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "=== 完成 ==="
Write-Host "  逐轮结果: scripts/results/ 下 <run_id>_stageC_<cfg>_seed<seed>.json"
Write-Host "  汇总: 脚本末尾已打印各配置跨 seed 的均值±std"
