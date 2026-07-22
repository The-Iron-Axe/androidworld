# Smoke-test 2 Hard tasks with Qwen3-VL-32B M3A baseline.
# Prerequisites:
#   1) Android emulator running (adb devices shows a device)
#   2) $env:OPENAI_API_KEY set
#   3) $env:OPENAI_BASE_URL set to full chat completions URL
#
# Example:
#   $env:OPENAI_BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"
#   $env:OPENAI_API_KEY = "sk-..."
#   .\scripts\run_hard_smoke.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not $env:OPENAI_API_KEY) {
  throw "OPENAI_API_KEY is not set."
}
if (-not $env:OPENAI_BASE_URL) {
  throw "OPENAI_BASE_URL is not set (need full .../v1/chat/completions URL)."
}

$extra = @()
if ($env:ADB_PATH) {
  $extra += "--adb_path=$env:ADB_PATH"
}

python run.py `
  --agent_name=m3a_qwen3_vl_32b `
  --difficulty=hard `
  --tasks=MarkorCreateNoteAndSms,SimpleCalendarAddOneEvent `
  --n_task_combinations=1 `
  --task_random_seed=30 `
  @extra
