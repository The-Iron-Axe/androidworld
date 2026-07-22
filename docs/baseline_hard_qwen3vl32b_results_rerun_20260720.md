# AndroidWorld Hard Baseline 结果（完整重跑）

## 配置

- **Agent**: m3a_qwen3_vl_32b (Qwen/Qwen3-VL-32B-Instruct)
- **框架**: M3A（截图 + UI + 每步两次 LLM 调用）
- **难度**: hard（共 19 个任务模板）
- **n_task_combinations**: 1
- **task_random_seed**: 30
- **结果目录**: `C:\Users\WRQ\android_world\runs\run_20260720T115444899496`
- **生成时间**: 2026-07-20 18:45:27
- **备注**: 本轮为完整重跑（新 checkpoint）。`MarkorMergeNotes` 首轮因 API 异常为 nan，已单独补跑后计入（失败 0.0，非异常）。

## 总览

| 指标 | 数值 |
|------|------|
| 已落盘任务 | 19 / 19 |
| 成功 (>0.5) | 4 |
| 部分成功 (0~0.5] | 0 |
| 失败 (=0) | 15 |
| 异常 (nan) | 0 |
| mean_success_rate（仅有效分数） | 0.211 (21.1%) |
| mean_success_rate（nan 按 0，共 19） | 0.211 (21.1%) |
| 总运行时间（各局 run_time 之和） | ~6.5 小时 |

## 分任务结果

| # | 任务 | is_successful | 步数 | 状态 | 耗时(分) |
|---|------|---------------|------|------|----------|
| 1 | `BrowserMultiply` | 0.0 | 18 | 失败 | 15.6 |
| 2 | `ExpenseAddMultipleFromGallery` | 0.0 | 11 | 失败 | 9.4 |
| 3 | `ExpenseAddMultipleFromMarkor` | 0.0 | 60 | 失败 | 45.9 |
| 4 | `ExpenseDeleteMultiple2` | 1.0 | 16 | 成功 | 11.1 |
| 5 | `MarkorCreateNoteAndSms` | 0.0 | 18 | 失败 | 16.7 |
| 6 | `MarkorMergeNotes` | 0.0 | 32 | 失败 | 35.4 |
| 7 | `MarkorTranscribeVideo` | 0.0 | 20 | 失败 | 15.0 |
| 8 | `OsmAndMarker` | 0.0 | 20 | 失败 | 14.7 |
| 9 | `OsmAndTrack` | 0.0 | 17 | 失败 | 11.2 |
| 10 | `RecipeAddMultipleRecipesFromImage` | 0.0 | 60 | 失败 | 51.2 |
| 11 | `RecipeAddMultipleRecipesFromMarkor` | 0.0 | 13 | 失败 | 9.6 |
| 12 | `RecipeAddMultipleRecipesFromMarkor2` | 0.0 | 60 | 失败 | 55.5 |
| 13 | `RecipeDeleteMultipleRecipesWithConstraint` | 1.0 | 4 | 成功 | 3.6 |
| 14 | `RetroSavePlaylist` | 1.0 | 21 | 成功 | 18.8 |
| 15 | `SaveCopyOfReceiptTaskEval` | 0.0 | 16 | 失败 | 12.5 |
| 16 | `SimpleCalendarAddOneEvent` | 1.0 | 17 | 成功 | 15.5 |
| 17 | `SportsTrackerActivitiesOnDate` | 0.0 | 20 | 失败 | 15.9 |
| 18 | `SportsTrackerTotalDistanceForCategoryOverInterval` | 0.0 | 20 | 失败 | 19.7 |
| 19 | `SportsTrackerTotalDurationForCategoryThisWeek` | 0.0 | 10 | 失败 | 10.7 |

## 成功任务

| 任务 | 步数 | 耗时(分) |
|------|------|----------|
| `ExpenseDeleteMultiple2` | 16 | 11.1 |
| `RecipeDeleteMultipleRecipesWithConstraint` | 4 | 3.6 |
| `RetroSavePlaylist` | 21 | 18.8 |
| `SimpleCalendarAddOneEvent` | 17 | 15.5 |

## 与上一份报告对比

上一份（`baseline_hard_qwen3vl32b_results.md`，run `run_20260717T103851876172`，含中途补跑）成功 3 / 15.8%。  
本轮完整重跑成功 **4 / 21.1%**；新增成功 `ExpenseDeleteMultiple2`；`MarkorMergeNotes` 本轮为正常失败而非 nan。

## 说明

- `is_successful`: `1.0` 成功，`0.0` 失败，中间值为复合任务部分分，`nan` 表示该局运行异常未正常打分。
- 正式 baseline 建议报告：**Hard 上 mean_success_rate = 21.1%**（19/19 均有有效分数）。
