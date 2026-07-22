# AndroidWorld Hard Baseline 结果（第三轮完整重跑）

## 配置

- **Agent**: m3a_qwen3_vl_32b (Qwen/Qwen3-VL-32B-Instruct)
- **框架**: M3A（截图 + UI + 每步两次 LLM 调用）
- **难度**: hard（共 19 个任务模板）
- **n_task_combinations**: 1
- **task_random_seed**: 30
- **结果目录**: `C:\Users\WRQ\android_world\runs\run_20260720T185932983508`
- **生成时间**: 2026-07-21 02:59:16
- **备注**: 本轮为第三遍完整重跑；中途曾中断，后用 checkpoint 续跑至 19/19。

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
| 总运行时间（各局 run_time 之和） | ~4.6 小时 |

## 分任务结果

| # | 任务 | is_successful | 步数 | 状态 | 耗时(分) |
|---|------|---------------|------|------|----------|
| 1 | `BrowserMultiply` | 0.0 | 17 | 失败 | 9.3 |
| 2 | `ExpenseAddMultipleFromGallery` | 0.0 | 9 | 失败 | 5.0 |
| 3 | `ExpenseAddMultipleFromMarkor` | 0.0 | 21 | 失败 | 13.0 |
| 4 | `ExpenseDeleteMultiple2` | 1.0 | 16 | 成功 | 8.3 |
| 5 | `MarkorCreateNoteAndSms` | 0.0 | 18 | 失败 | 9.5 |
| 6 | `MarkorMergeNotes` | 0.0 | 78 | 失败 | 37.1 |
| 7 | `MarkorTranscribeVideo` | 0.0 | 20 | 失败 | 12.9 |
| 8 | `OsmAndMarker` | 0.0 | 8 | 失败 | 5.0 |
| 9 | `OsmAndTrack` | 0.0 | 27 | 失败 | 17.2 |
| 10 | `RecipeAddMultipleRecipesFromImage` | 0.0 | 60 | 失败 | 43.8 |
| 11 | `RecipeAddMultipleRecipesFromMarkor` | 0.0 | 60 | 失败 | 37.9 |
| 12 | `RecipeAddMultipleRecipesFromMarkor2` | 0.0 | 60 | 失败 | 24.9 |
| 13 | `RecipeDeleteMultipleRecipesWithConstraint` | 1.0 | 4 | 成功 | 1.6 |
| 14 | `RetroSavePlaylist` | 1.0 | 49 | 成功 | 19.9 |
| 15 | `SaveCopyOfReceiptTaskEval` | 0.0 | 16 | 失败 | 7.0 |
| 16 | `SimpleCalendarAddOneEvent` | 1.0 | 16 | 成功 | 8.2 |
| 17 | `SportsTrackerActivitiesOnDate` | 0.0 | 20 | 失败 | 9.9 |
| 18 | `SportsTrackerTotalDistanceForCategoryOverInterval` | 0.0 | 3 | 失败 | 1.6 |
| 19 | `SportsTrackerTotalDurationForCategoryThisWeek` | 0.0 | 10 | 失败 | 5.6 |

## 成功任务

| 任务 | 步数 | 耗时(分) |
|------|------|----------|
| `ExpenseDeleteMultiple2` | 16 | 8.3 |
| `RecipeDeleteMultipleRecipesWithConstraint` | 4 | 1.6 |
| `RetroSavePlaylist` | 49 | 19.9 |
| `SimpleCalendarAddOneEvent` | 16 | 8.2 |

## 三轮对比

| 轮次 | 结果目录 | 成功 | mean | 备注 |
|------|----------|------|------|------|
| 1（补跑版） | `run_20260717T103851876172` | 3 | 15.8% | 含 FTS/ffmpeg 补跑 |
| 2 | `run_20260720T115444899496` | 4 | 21.1% | 完整重跑 |
| 3（本轮） | `run_20260720T185932983508` | 4 | 21.1% | 与第 2 轮一致 |

**三轮均成功：** `RecipeDeleteMultipleRecipesWithConstraint`、`RetroSavePlaylist`、`SimpleCalendarAddOneEvent`  
**后两轮均成功：** `ExpenseDeleteMultiple2`（第 1 轮失败）

## 说明

- `is_successful`: `1.0` 成功，`0.0` 失败，中间值为复合任务部分分，`nan` 表示该局运行异常未正常打分。
- 正式 baseline 建议报告：**Hard 上 mean_success_rate ≈ 21.1%**（第 2、3 轮一致；第 1 轮因早期环境问题偏低）。
