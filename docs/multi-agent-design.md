# 多智能体系统交接文档

## 1. 设计原则

Reflector 不按照“查看多长时间的历史”划分，而按照“检查什么对象”划分。
    原划分：
    单步反思 → 轨迹反思 → 全局反思

    当前划分：
    动作核验 → 进度审计 → 证据认证

三个子模块分别回答三个不同性质的问题：

* **Action Verifier**：这一步动作是否按声明执行正确？
* **Progress Auditor**：当前执行是否真正接近任务目标？
* **Evidence Certifier**：最终结果是否逐项满足要求，并有有效证据？

* * *

## 2. 系统层级

    Multi-Agent System
    │
    ├── Planner
    │
    ├── Executor
    │
    └── Reflector
        ├── Action Verifier
        ├── Progress Auditor
        └── Evidence Certifier

| 子模块                | 中文名称 | 检查对象 | 核心问题        |
| ------------------ | ---- | ---- | ----------- |
| Action Verifier    | 动作核验 | 当前动作 | 手有没有做对？     |
| Progress Auditor   | 进度审计 | 目标进度 | 是否真的在向目标推进？ |
| Evidence Certifier | 证据认证 | 最终交付 | 是否逐项合格并有证据？ |

三个子模块的区别不在于调用时间，而在于验证责任。

* * *

## 3. Planner

Planner 负责维护任务目标、子目标、进度条件和验收标准。

任务开始时，Planner 需要生成三类内容。

### 3.1 Subgoal Plan

将用户任务拆分为具有依赖关系的子目标。

示例：
    任务：创建一个明天早上 7 点的闹钟

    子目标 1：进入新建闹钟界面
    子目标 2：将时间设置为 07:00
    子目标 3：将日期设置为明天
    子目标 4：启用闹钟
    子目标 5：保存闹钟

Planner 可以在执行过程中根据已验证状态刷新剩余子目标。

### 3.2 Progress Conditions

定义哪些状态表示任务取得了真实进展。
    P1：已经进入闹钟编辑界面
    P2：时间已经设置为 07:00
    P3：日期已经设置为明天
    P4：启用开关已经打开
    P5：闹钟已经保存并出现在列表中

这些条件由 Progress Auditor 检查和更新。

### 3.3 Acceptance Checklist

定义最终交付必须满足的验收项。
    A1：闹钟时间等于 07:00
    A2：闹钟日期等于明天
    A3：闹钟处于启用状态
    A4：闹钟已经成功保存

每个验收项需要明确：

* 检查对象；
* 期望值；
* 是否为必要项；
* 可接受的证据形式；
* 与该结论冲突的状态。

Planner 负责定义验收标准，但不负责认证标准是否满足。

* * *

## 4. Executor

Executor 负责当前子目标下的具体动作决策。

每次执行前，Executor 必须提交 **Action Claim**，即对本次动作效果作出事前声明。

### 4.1 Action Claim

Action Claim 至少包括：
    动作：
    点击当前闹钟编辑页中的时间区域

    动作意图：
    打开时间设置界面

    预期结果：
    屏幕上出现时间选择框

Action Claim 必须在动作执行前产生，不能根据执行结果倒推或修改。

Executor 只负责：

* 决定当前动作；
* 指定操作目标；
* 声明动作意图；
* 声明预期结果；
* 提出子目标或任务完成申请。

Executor 不负责认证动作成功、进度增长或任务完成。

* * *

# 5. Action Verifier

## 5.1 职责

Action Verifier 只检查：

> Executor 在 Action Claim 中声明的动作效果是否实际发生？

例如 Executor 声明：
    点击时间区域，是为了打开时间选择框。

Action Verifier 只核实：
    时间选择框是否真的出现？

它不判断当前路线是否合理，也不判断整个任务是否完成。

* * *

## 5.2 输入

Action Verifier 接收：

* Action Claim；
* 动作前界面状态；
* 动作后界面状态；
* 实际执行动作；
* 被操作目标的界面信息。

* * *

## 5.3 确定性检查优先

Action Verifier 采用两阶段核验：
    确定性检查
    → 无法明确裁决
    → AI 语义核验

优先直接检查：

* 指定元素是否出现或消失；
* 页面标识是否改变；
* 文本值是否更新；
* 输入内容是否正确；
* 开关状态是否改变；
* 弹窗是否出现；
* 点击位置是否落在目标元素范围内；
* 是否出现明确错误信息。

例如，预期结果是“时间选择框出现”，可以检查：
    小时选择组件是否出现
    分钟选择组件是否出现
    确认按钮是否出现

这些条件已经足以确认时，不再调用复杂模型。

只有在页面结构难以解析、视觉含义存在歧义或元素发生较大变化时，才调用 AI 进行语义判断。

* * *

## 5.4 输出

Action Verifier 输出三类主要结果。

### `CORRECT`

动作兑现了 Action Claim。
    结论：CORRECT
    证据：时间选择框已经出现。

### `MISGROUNDED`

动作作用到了错误对象，或者产生了错误页面转移。
    结论：MISGROUNDED
    证据：点击后打开了日期设置页面，而非时间选择框。

### `NO_EFFECT`

动作未产生声明中的预期结果，也没有产生其他有效变化。
    结论：NO_EFFECT
    证据：动作前后目标区域和页面状态均未发生变化。

Action Verifier 的结论只描述动作结果，不生成高层计划。

* * *

# 6. Progress Auditor

## 6.1 职责

Progress Auditor 只检查：

> 当前任务状态是否比之前更接近用户目标？

它不把屏幕变化直接视为任务进展，也不以动作是否重复作为主要判定依据。

例如：
    时区页面
    → 日历页面
    → 铃声页面
    → 闹钟首页

虽然屏幕持续变化，但如果闹钟时间、日期、开关和保存状态均未改变，就没有产生真实进度。

* * *

## 6.2 Progress Ledger

Progress Auditor 维护一份 **Progress Ledger**。
    P1：进入闹钟编辑页面        已满足
    P2：时间设置为 07:00        已满足
    P3：日期设置为明天          未满足
    P4：启用开关打开            未满足
    P5：闹钟保存成功            未满足

每次检查时，对比动作前后的目标条件：
    动作前满足：P1、P2
    动作后满足：P1、P2、P3

    进度增量：+1

真实进展包括：

* 新的 Progress Condition 被满足；
* 某一条件从部分满足变为完全满足；
* 与目标冲突的状态被解除；
* 尚未解决的必要条件减少；
* 进入完成目标所必需的前置状态。

* * *

## 6.3 假忙活

页面不断变化，但 Progress Ledger 没有新增满足项时，判定为假忙活。
    打开时区页面
    → 返回
    → 打开铃声页面
    → 返回
    → 打开重复设置页面

如果任务条件始终是：
    时间仍不是 07:00
    日期仍不是明天
    启用开关仍然关闭

则输出：
    BUSY_WITHOUT_PROGRESS

假忙活不要求动作完全重复，也不要求页面截图相同。

* * *

## 6.4 原地绕圈

原地绕圈依据的是**目标进度状态反复相同**。
    状态 1：P1 已满足，P2-P5 未满足
    状态 2：P1 已满足，P2-P5 未满足
    状态 3：P1 已满足，P2-P5 未满足

即使三个状态对应不同页面，只要任务条件长期没有变化，就可能属于无效循环。

重复动作、重复截图和相似页面可以作为辅助信号，但不能代替目标进度判断。

* * *

## 6.5 输出

Progress Auditor 输出四类结果：

### `ADVANCING`

新增目标条件被满足，任务正在推进。

### `STALLED`

近期没有产生新的有效进展。

### `LOOPING`

系统反复回到相同或等价的进度状态。

### `BUSY_WITHOUT_PROGRESS`

动作和页面持续变化，但目标条件长期没有增长。

示例：
    结论：BUSY_WITHOUT_PROGRESS

    当前已满足：
    P1：进入闹钟编辑页面

    持续未满足：
    P2：时间设置为 07:00
    P3：日期设置为明天

    证据：
    最近三个状态均未改变任何目标条件。

Progress Auditor 只负责判断进度，不负责修改计划。

* * *

# 7. Evidence Certifier

## 7.1 职责

Evidence Certifier 只检查：

> Acceptance Checklist 中的每条要求是否满足，并且是否存在足以支持该结论的有效证据？

它不通过阅读长操作历史来猜测任务是否完成，而是逐项认证。
    验收项
    → 检查实际值
    → 查找对应证据
    → 验证证据有效性
    → PASS / FAIL

* * *

## 7.2 Evidence Bundle

每个验收项必须绑定一份 **Evidence Bundle**。

Evidence Bundle 包括：

* 验收项编号；
* 验收要求；
* 实际观察值；
* 证据截图；
* 截图对应步骤或时间；
* 证据所在区域；
* 证据是否仍然有效；
* 最终认证结果。

示例：
    验收项：A1
    要求：闹钟时间等于 07:00

    实际值：07:00
    证据：步骤 12 的最终闹钟列表截图
    证据区域：列表第一项的时间文本
    结论：PASS

失败示例：
    验收项：A3
    要求：闹钟处于启用状态

    实际值：开关关闭
    证据：步骤 12 的最终闹钟列表截图
    结论：FAIL

* * *

## 7.3 无证据不认证

Evidence Certifier 使用严格规则：
    执行过相关动作
    ≠ 最终状态满足要求
    ≠ 有证据证明完成

以下情况不能通过认证：

* 只有 Executor 的完成声明；
* 只有动作历史；
* 只有“曾经点击过保存”；
* 只有中间页面状态；
* 缺少对应截图；
* 截图无法看清关键值；
* 证据与验收项没有直接关系。

统一输出：
    NO_EVIDENCE

示例：
    验收项：A2
    要求：日期设置为明天

    历史记录：
    Executor 曾点击日期设置按钮

    有效证据：
    缺失

    结论：FAIL
    原因：没有证据证明最终日期确实为明天。

* * *

## 7.4 证据新鲜度

证据必须能够代表最终交付状态。

例如：
    步骤 6：闹钟开关打开
    步骤 9：闹钟开关被关闭

步骤 6 的截图不能继续用于证明最终开关为打开状态。

Evidence Certifier 需要检查：

* 证据之后是否执行过可能改变该值的动作；
* 当前状态是否与历史证据冲突；
* 不同 Evidence Bundle 是否互相矛盾；
* 证据是否来自保存后的最终状态；
* 证据中的目标对象是否确实属于本次任务。

存在冲突时，该验收项不能直接通过，需要重新获取当前状态证据。

* * *

## 7.5 最终输出

Evidence Certifier 必须逐项输出，而不是只给整体结论。
    A1：时间等于 07:00
    PASS
    证据：最终闹钟列表显示 07:00

    A2：日期等于明天
    PASS
    证据：闹钟详情页面显示明天

    A3：闹钟处于启用状态
    FAIL
    证据：最终列表中的开关处于关闭状态

    A4：闹钟已保存
    PASS
    证据：目标闹钟已出现在列表中

整体结果：
    FINAL RESULT：FAIL

    未通过项：
    A3：闹钟未启用

只有所有必要验收项均为 `PASS` 时，任务才能被认证为：
    FINAL RESULT：PASS

* * *

# 8. 三类核心记录

| 记录              | 产生者                            | 主要使用者            | 作用            |
| --------------- | ------------------------------ | ---------------- | ------------- |
| Action Claim    | Executor                       | Action Verifier  | 声明当前动作应产生什么结果 |
| Progress Ledger | Planner 定义，Progress Auditor 更新 | Planner、Executor | 表示任务是否真正取得进展  |
| Evidence Bundle | Evidence Certifier             | 最终验收流程           | 证明每条交付要求已经满足  |

三者不能互相替代：
    Action Claim 不能证明任务完成。

    Progress Ledger 不能证明某次点击准确。

    动作历史不能替代最终验收证据。

* * *

# 9. 完整运行流程

## 9.1 初始化

    Planner
    → 生成 Subgoal Plan
    → 生成 Progress Conditions
    → 生成 Acceptance Checklist

## 9.2 单步执行

    Executor
    → 提交 Action Claim
    → 执行动作

## 9.3 动作核验

    Action Verifier
    → 优先执行确定性检查
    → 无法裁决时调用 AI
    → 输出 CORRECT / MISGROUNDED / NO_EFFECT

## 9.4 进度审计

    Progress Auditor
    → 更新 Progress Ledger
    → 计算目标条件增量
    → 输出 ADVANCING / STALLED / LOOPING / BUSY_WITHOUT_PROGRESS

## 9.5 证据收集

当某个验收条件出现时：
    Evidence Certifier
    → 判断当前状态能否证明验收项
    → 保存截图和证据区域
    → 生成或更新 Evidence Bundle

## 9.6 最终认证

    Executor 提出任务完成
    → Evidence Certifier 逐项检查 Acceptance Checklist
    → 验证证据完整性、新鲜度和一致性
    → 输出 FINAL RESULT

未通过时，将缺失项和证据问题提交给 Planner，由 Planner调整后续子目标。

* * *

# 10. 职责边界

## Planner

负责：

* 拆分任务；
* 定义进度条件；
* 定义验收清单；
* 刷新和修复计划。

不负责：

* 判断动作是否点对；
* 认证最终证据。

## Executor

负责：

* 选择动作；
* 执行动作；
* 事前提交 Action Claim。

不负责：

* 自行确认动作成功；
* 自行确认任务完成。

## Action Verifier

负责：

* 验证 Action Claim 是否兑现；
* 区分做对、点偏和无效果。

不负责：

* 判断整体执行方向；
* 判断最终交付。

## Progress Auditor

负责：

* 检查目标条件是否增长；
* 识别停滞、绕圈和假忙活。

不负责：

* 判断具体点击坐标是否正确；
* 认证最终交付。

## Evidence Certifier

负责：

* 按 Acceptance Checklist 逐项核验；
* 为验收项绑定有效证据；
* 检查证据新鲜度与一致性；
* 输出 PASS、FAIL 或 NO_EVIDENCE。

不负责：

* 选择下一步动作；
* 修改任务计划。

* * *

# 11. 最终职责总结

    Action Verifier
    查动作有没有按声明做对。
    
    Progress Auditor
    查任务条件有没有真正向前推进。
    
    Evidence Certifier
    查最终交付是否逐项合格，并认证对应证据。

一句话概括：

> Reflector 由三个不同工种组成，分别检查动作、进度和证据，而不是由同一个角色查看不同时长的历史。

* * *

# 12. 实现附录（v1.0，2026-08-06）

## 12.1 代码结构

多智能体层是**独立的 agent 子类**，与记忆轴正交。U1-U4 记忆模块和 `memory_agent.py` 一行不改。

```
android_world/agents/
  multi_agent.py          # MultiAgentReflectorAgent + 数据类（ActionClaim/ProgressLedger/…）
  multi_agent_verifier.py # 三个纯函数验证器（LLM 驱动）
  multi_agent_test.py     # 单元 + 集成测试（20 个）
run.py                    # --multiagent flag
scripts/ablation_hierarchical.py  # --multiagent flag 透传
```

## 12.2 类映射

| 设计文档角色 | 实现 |
| --- | --- |
| Planner | `MultiAgentReflectorAgent._planner_plan()`（开局一次）+ `_planner_replan()`（STALLED 时，`MAX_REPLANS=2` 上限） |
| Executor | 继承的 `M3A.step()` 决策路径（不改）；System 从 `action_reason` 提取 `ActionClaim` |
| Reflector | `multi_agent_verifier.py` 三个纯函数：`verify_action` / `audit_progress` / `certify_evidence` |

## 12.3 Hook 插入表

所有覆写 hook 首行 `if not self._multiagent:` 门控，关闭时行为与现有记忆 agent 完全一致。

| Hook | 多智能体行为 |
| --- | --- |
| `_build_action_prompt` | 首步 `_planner_plan` 一次 → 写 U1.pending/current_subgoal → 注入 `## Plan` + 可选 `## Verifier Feedback` |
| `_accept_task_done` | 刷新后的最终 UI 上跑全局 Evidence Certifier；PASS 才允许 `done=True`；FAIL → 反馈缺口 + `_planner_replan`，episode 继续 |
| `_on_step_complete` | 不调 super()；replay 步跳过；Action Verifier 门控 U3 画边；非 CORRECT 写入反馈并计入 stall；Progress Auditor + 子目标级 Evidence Certifier 定序；U1 账本复刻 |
| `_on_task_done` | 仅在 accept 之后：super() (U2 缓冲)；`_certified` 已在 `_accept_task_done` 设定 |
| `flush_memory` | max_steps 耗尽路径补跑全局认证 |
| `set_episode_success` | `success = success and self._certified`（内部认证与外部真值并存） |
| `reset` | 清多智能体状态与 reflector 反馈 |

## 12.4 三个验证器信号清单

| Verifier | 输入 | 输出 |
| --- | --- | --- |
| Action | claim + 前后元素 + 前后截图 | CORRECT / MISGROUNDED / NO_EFFECT |
| Progress | 台账签名 + 当前状态 | ADVANCING / STALLED / LOOPING / BUSY_WITHOUT_PROGRESS |
| Evidence | 验收清单 + 最终状态 | 逐项 PASS/FAIL/NO_EVIDENCE + 整体 |

## 12.5 Flag 矩阵（2×2 消融）

| 命令 | 结构 |
| --- | --- |
| `m3a_qwen3_vl_32b` | baseline：单智能体无记忆 |
| `m3a_qwen3_vl_32b_mem --u1 --u2 --u3 --u4` | 单智能体 + 记忆 |
| `m3a_qwen3_vl_32b_mem --multiagent` | 多智能体无记忆 |
| `m3a_qwen3_vl_32b_mem --multiagent --u1 --u2 --u3 --u4` | 多智能体 + 记忆（最终系统） |

## 12.6 关键实现约束

1. **U3 画边门控**：只有 Action Verifier 判 CORRECT 才 `u3.record_transition`，MISGROUNDED/NO_EFFECT 不画边。
2. **replay 步跳过**：U2 确定性回放的 step 无 action_reason，跳过全部 verifier。
3. **子目标认证定序**：`Progress ADVANCING → Evidence(子目标) 认证 → 通过才推进台账/换子目标`。
4. **证据新鲜度**：证据后被修改该值 → 该证据失效（触发重新获取）。
5. **ActionClaim frozen**：事前从 `action_reason` 提取，任何 verifier 不可篡改（§4.1）。
6. **非 UI 动作跳过**：status/answer/wait 无 target，跳过 Action Verifier（默认 CORRECT）。

