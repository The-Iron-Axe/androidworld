# Hard Baseline 步数 vs 最优步数 / complexity 对比

对照来源：

- Baseline：`docs/baseline_hard_qwen3vl32b_results.md`  
（`m3a_qwen3_vl_32b`，run `run_20260717T103851876172`，mean SR 15.8%）
- 最优步数：`android_world/task_metadata.json` 的 `optimal_steps`
- complexity：各任务类上的 `complexity` 字段；步数上限 `max_steps = int(10 × complexity)`（`suite_utils._allocate_step_budget`）

说明：

- **实际步数**：该次 episode 的 `episode_length`
- **最优步数**：官方标注的参考最短路径
- **complexity / 上限**：评测允许的最大步数预算
- **倍率(实际/最优)**：相对最优路径的膨胀；成功任务更有参考价值
- **占用上限**：实际 / 上限；接近 1.0 表示基本跑满预算

## 总览


| 指标                 | 数值                                                                                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hard 任务数           | 19                                                                                                                                                                  |
| Baseline 成功        | 3 / 19 (15.8%)                                                                                                                                                      |
| 最优步数合计             | 325                                                                                                                                                                 |
| 实际步数合计             | 446                                                                                                                                                                 |
| 上限步数合计             | 810                                                                                                                                                                 |
| 全体实际/最优            | 1.37×                                                                                                                                                               |
| 全体实际/上限            | 0.55×                                                                                                                                                               |
| 仅成功任务实际/最优         | 60/62 ≈ **0.97×**                                                                                                                                                   |
| 仅失败任务实际/最优         | 386/263 ≈ **1.47×**                                                                                                                                                 |
| 跑满或接近上限（实际 ≥ 上限−2） | `MarkorCreateNoteAndSms`(18/18)、`MarkorTranscribeVideo`(20/20)、`OsmAndMarker`(20/20)、`RecipeAddMultipleRecipesFromMarkor`(60/60)、`SaveCopyOfReceiptTaskEval`(16/16) |




## 分任务对比


| #   | 任务                                                  | 结果     | complexity | 上限 (×10) | 最优  | 实际  | 实际/最优     | 实际/上限    |
| --- | --------------------------------------------------- | ------ | ---------- | -------- | --- | --- | --------- | -------- |
| 1   | `BrowserMultiply`                                   | 失败     | 2.2        | 22       | 11  | 19  | 1.73×     | 0.86     |
| 2   | `ExpenseAddMultipleFromGallery`                     | 失败     | 6          | 60       | 10  | 32  | 3.20×     | 0.53     |
| 3   | `ExpenseAddMultipleFromMarkor`                      | 失败     | 6          | 60       | 15  | 19  | 1.27×     | 0.32     |
| 4   | `ExpenseDeleteMultiple2`                            | 失败     | 3.4        | 34       | 17  | 28  | 1.65×     | 0.82     |
| 5   | `MarkorCreateNoteAndSms`                            | 失败     | 1.8        | 18       | 9   | 18  | 2.00×     | **1.00** |
| 6   | `MarkorMergeNotes`                                  | 失败     | 7.8        | 78       | 39  | 33  | 0.85×     | 0.42     |
| 7   | `MarkorTranscribeVideo`                             | 失败     | 2          | 20       | 10  | 20  | 2.00×     | **1.00** |
| 8   | `OsmAndMarker`                                      | 失败     | 2.0        | 20       | 10  | 20  | 2.00×     | **1.00** |
| 9   | `OsmAndTrack`                                       | 失败     | 12         | 120      | 60  | 22  | 0.37×     | 0.18     |
| 10  | `RecipeAddMultipleRecipesFromImage`                 | 失败     | 6          | 60       | 13  | 35  | 2.69×     | 0.58     |
| 11  | `RecipeAddMultipleRecipesFromMarkor`                | 失败     | 6          | 60       | 24  | 60  | 2.50×     | **1.00** |
| 12  | `RecipeAddMultipleRecipesFromMarkor2`               | 失败     | 6          | 60       | 26  | 27  | 1.04×     | 0.45     |
| 13  | `RecipeDeleteMultipleRecipesWithConstraint`         | **成功** | 4          | 40       | 20  | 4   | **0.20×** | 0.10     |
| 14  | `RetroSavePlaylist`                                 | **成功** | 5          | 50       | 25  | 39  | **1.56×** | 0.78     |
| 15  | `SaveCopyOfReceiptTaskEval`                         | 失败     | 1.6        | 16       | 8   | 16  | 2.00×     | **1.00** |
| 16  | `SimpleCalendarAddOneEvent`                         | **成功** | 3.4        | 34       | 17  | 17  | **1.00×** | 0.50     |
| 17  | `SportsTrackerActivitiesOnDate`                     | 失败     | 2          | 20       | 10  | 3   | 0.30×     | 0.15     |
| 18  | `SportsTrackerTotalDistanceForCategoryOverInterval` | 失败     | 2.2        | 22       | 11  | 4   | 0.36×     | 0.18     |
| 19  | `SportsTrackerTotalDurationForCategoryThisWeek`     | 失败     | 1.6        | 16       | 8   | 10  | 1.25×     | 0.62     |


规律：多数 Hard 任务满足 **最优 ≈ 上限 / 2**（即 `optimal_steps ≈ 5 × complexity`）。

## 成功任务效率


| 任务                                          | complexity | 上限  | 最优  | 实际  | 解读           |
| ------------------------------------------- | ---------- | --- | --- | --- | ------------ |
| `SimpleCalendarAddOneEvent`                 | 3.4        | 34  | 17  | 17  | 与最优持平，用掉一半预算 |
| `RecipeDeleteMultipleRecipesWithConstraint` | 4          | 40  | 20  | 4   | 远快于最优与预算     |
| `RetroSavePlaylist`                         | 5          | 50  | 25  | 39  | 成功但超最优，仍未触顶  |




## 失败但仍「步数少于最优」的题目

通常是提前 `done` / 很快停，不是更优完成：


| 任务                                                  | complexity | 上限  | 最优  | 实际  | 结果  |
| --------------------------------------------------- | ---------- | --- | --- | --- | --- |
| `MarkorMergeNotes`                                  | 7.8        | 78  | 39  | 33  | 失败  |
| `OsmAndTrack`                                       | 12         | 120 | 60  | 22  | 失败  |
| `SportsTrackerActivitiesOnDate`                     | 2          | 20  | 10  | 3   | 失败  |
| `SportsTrackerTotalDistanceForCategoryOverInterval` | 2.2        | 22  | 11  | 4   | 失败  |




## 浪费最多（实际/最优高，且失败）


| 任务                                   | complexity | 上限  | 最优  | 实际  | 实际/最优 | 是否触顶  |
| ------------------------------------ | ---------- | --- | --- | --- | ----- | ----- |
| `ExpenseAddMultipleFromGallery`      | 6          | 60  | 10  | 32  | 3.20× | 否     |
| `RecipeAddMultipleRecipesFromImage`  | 6          | 60  | 13  | 35  | 2.69× | 否     |
| `RecipeAddMultipleRecipesFromMarkor` | 6          | 60  | 24  | 60  | 2.50× | **是** |
| `MarkorCreateNoteAndSms`             | 1.8        | 18  | 9   | 18  | 2.00× | **是** |
| `MarkorTranscribeVideo`              | 2          | 20  | 10  | 20  | 2.00× | **是** |
| `OsmAndMarker`                       | 2.0        | 20  | 10  | 20  | 2.00× | **是** |
| `SaveCopyOfReceiptTaskEval`          | 1.6        | 16  | 8   | 16  | 2.00× | **是** |




## 备注

- `complexity` 定义在各 `TaskEval` 子类（如 `recipe.py`、`expense.py`），不是 `task_metadata.json`。
- `difficulty`（easy/medium/hard）在 `task_metadata.json`；`complexity` 只负责动态分配步数预算。
- 本表仅覆盖 Hard baseline 的 19 题。

