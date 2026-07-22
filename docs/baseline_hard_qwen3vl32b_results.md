# AndroidWorld Hard Baseline 结果

## 配置

- **Agent**: m3a_qwen3_vl_32b (Qwen/Qwen3-VL-32B-Instruct)
- **框架**: M3A（截图 + UI + 每步两次 LLM 调用）
- **难度**: hard（共 19 个任务模板）
- **n_task_combinations**: 1
- **结果目录**: `C:\Users\WRQ\android_world\runs\run_20260717T103851876172`
- **生成时间**: 2026-07-20 11:39:13
- **备注**: 主机缺 FTS4 时已用设备端 sqlite3 回退；`RetroSavePlaylist` 依赖本机 ffmpeg，装好后重跑成功。

## 总览


| 指标                              | 数值            |
| ------------------------------- | ------------- |
| 已落盘任务                           | 19 / 19       |
| 成功 (>0.5)                       | 3             |
| 部分成功 (0~0.5]                    | 0             |
| 失败 (=0)                         | 16            |
| 异常 (nan)                        | 0             |
| mean_success_rate（仅有效分数）        | 0.158 (15.8%) |
| mean_success_rate（nan 按 0，共 19） | 0.158 (15.8%) |
| 总运行时间（各局 run_time 之和）           | ~4.5 小时       |




## 分任务结果


| #   | 任务                                                  | is_successful | 步数  | 状态  | 耗时(分) |
| --- | --------------------------------------------------- | ------------- | --- | --- | ----- |
| 1   | BrowserMultiply                                     | 0.0           | 19  | 失败  | 17.0  |
| 2   | ExpenseAddMultipleFromGallery                       | 0.0           | 32  | 失败  | 20.2  |
| 3   | ExpenseAddMultipleFromMarkor                        | 0.0           | 19  | 失败  | 10.7  |
| 4   | ExpenseDeleteMultiple2                              | 0.0           | 28  | 失败  | 15.3  |
| 5   | MarkorCreateNoteAndSms                              | 0.0           | 18  | 失败  | 10.2  |
| 6   | MarkorMergeNotes                                    | 0.0           | 33  | 失败  | 17.5  |
| 7   | MarkorTranscribeVideo                               | 0.0           | 20  | 失败  | 9.8   |
| 8   | OsmAndMarker                                        | 0.0           | 20  | 失败  | 9.6   |
| 9   | OsmAndTrack                                         | 0.0           | 22  | 失败  | 25.5* |
| 10  | RecipeAddMultipleRecipesFromImage                   | 0.0           | 35  | 失败  | 22.9  |
| 11  | RecipeAddMultipleRecipesFromMarkor                  | 0.0           | 60  | 失败  | 31.0  |
| 12  | RecipeAddMultipleRecipesFromMarkor2                 | 0.0           | 27  | 失败  | 18.2  |
| 13  | RecipeDeleteMultipleRecipesWithConstraint           | 1.0           | 4   | 成功  | 4.2   |
| 14  | RetroSavePlaylist                                   | 1.0           | 39  | 成功  | 22.2  |
| 15  | SaveCopyOfReceiptTaskEval                           | 0.0           | 16  | 失败  | 13.0  |
| 16  | `SimpleCalendarAddOneEvent`                         | 1.0           | 17  | 成功  | 7.8   |
| 17  | `SportsTrackerActivitiesOnDate`                     | 0.0           | 3   | 失败  | 5.2   |
| 18  | `SportsTrackerTotalDistanceForCategoryOverInterval` | 0.0           | 4   | 失败  | 1.9   |
| 19  | `SportsTrackerTotalDurationForCategoryThisWeek`     | 0.0           | 10  | 失败  | 5.8   |


 `OsmAndTrack` checkpoint 中 `run_time` 曾异常偏大（含挂起/跨会话），表中耗时沿用首次完整评测约 25.5 分钟；步数以最新 checkpoint（22）为准。总运行时间按调整后约 4.5 小时计。

## 成功任务


| 任务                                          | 步数  | 耗时(分) |
| ------------------------------------------- | --- | ----- |
| `SimpleCalendarAddOneEvent`                 | 17  | 7.8   |
| `RecipeDeleteMultipleRecipesWithConstraint` | 4   | 4.2   |
| `RetroSavePlaylist`                         | 39  | 22.2  |




## 说明

- `is_successful`: `1.0` 成功，`0.0` 失败，中间值为复合任务部分分，`nan` 表示该局运行异常未正常打分。
- 正式 baseline 建议报告：**Hard 上 mean_success_rate = 15.8%**（19/19 均有有效分数）。
- 相对首轮完整跑：成功 1→3，异常 5→0。

