# 记忆系统设计文档（U1-U5）

> 版本：v1.0 · 2026-08-06
> 定位：纯设计文档，不含实现代码。与 `docs/multi-agent-design.md`（多智能体 v2.1）配套。

---

## 1. 总纲

### 1.1 核心原则

**记忆机制不是 Agent。** U1-U5 是纯数据层——无 LLM 调用、无环境交互、无 agent 继承。Planner-Executor-Reflector 才是 agent 层。二者正交：先设计记忆，后设计多智能体。

**U1-U4 在线、U5 离线。** U1-U4 在 agent 运行时读写（显式记忆），U5 在任务跑完后离线微调（把记忆使用能力内化进模型权重）。

**U1-U5 是正交维度，不是层级叠加。** 每个都是独立开关、独立可验证、独立可消融。消融协议的设计目的就是检测哪层真有用，防止堆料。

### 1.2 记忆分类学（源自综述）

| 层   | 名称      | 主要保存什么                        | 最简单的问题        |
| --- | ------- | ----------------------------- | ------------- |
| U1  | 任务状态记忆  | 当前任务的动作历史、进度、子目标和约束状态         | 现在做到哪了？       |
| U2  | 情景轨迹记忆  | 过去具体任务的轨迹、关键片段和执行案例           | 以前怎么做过？       |
| U3  | 环境知识记忆  | 应用功能、页面结构、控件语义和状态转移知识         | 这个环境怎么运作？     |
| U4  | 程序性技能记忆 | 从多个经验中抽象或验证得到的参数化工作流、技能和可执行程序 | 这类任务应该调用什么程序？ |
| U5  | 内部化记忆   | 模型或控制器学到的记忆使用、检索和更新能力         | 模型自身已经学会什么？   |

---

## 2. U1 任务状态记忆

### 2.1 定位

最轻量的一层。维护当前任务的**结构化执行状态**，不是历史经验。

### 2.2 数据结构

```
TaskState:
  goal            任务目标
  current_app     当前 app 包名
  current_page    当前页面/Activity 描述
  current_subgoal 当前子目标
  completed       已完成的子目标列表
  pending         待办子目标队列（头部 = 当前）
  constraints     执行中发现的约束（如"日期格式必须 YYYY-MM-DD"）
  last_action     最近一次动作
  last_effect     最近动作的效果摘要
  failure_count   当前子目标连续失败次数
  step_count      总步数
```

### 2.3 读写

- **写**：agent 每步成功后更新（app/页面/动作/效果/失败计数）。
- **读**：注入 action prompt 的 `## Task State (U1)` 块，供 Executor 感知当前进度。

### 2.4 消融开关

`--u1` 独立开关。

---

## 3. U2 情景轨迹记忆

### 3.1 定位

跨任务的经验复用。包装 Darwinian Memory System（DMS, arXiv:2601.22528）作为纯数据模块。

### 3.2 数据结构

- 单元：`MemoryEntry`，按 `Plan(precondition, goal)` 索引。
- 轨迹：`ObsAct(observation, action, step_index)` 序列，action 为 JSONAction。

### 3.3 核心机制（源自 DMS）

| 机制             | 作用                                                |
| -------------- | ------------------------------------------------- |
| 双因子检索          | `Score = sim(pre) × sim(goal)`，起始状态 + 目标都匹配       |
| ε-Mutation     | 概率 ε 强制重探索，发现更短轨迹则覆盖                              |
| 生存值            | `S = Utility × AdaptiveDecay × Reliability`，多因子打分 |
| Elbow 剪枝       | 自动容量调节                                            |
| 贝叶斯风险门控        | 动态阈值 T_global，失败率高时收紧检索                           |
| K-Verification | 连续 K 次失败触发淘汰                                      |

### 3.4 子计划粒度

- 提供 `add_sub_plan` / `retrieve_sub_plan_hint` / `retrieve_sub_plan_replay` 子计划粒度 API，对齐 `Plan(precondition, goal)`。
- **Planner 的子计划分解是 agent 层职责，U2 只存和检索**，不生成子计划。

### 3.5 消融开关

`--u2` 独立开关。

---

## 4. U3 环境知识记忆

### 4.1 定位

环境运作方式：页面结构 + 状态转移。忠实于 PG-Agent（arXiv:2509.03536）§3.1。

### 4.2 数据结构

```
PageNode:  page_id, page_summary（语义身份）, app
PageEdge:  source --action--> target, task（该动作服务过的任务）, count
```

页面按 embedding 余弦相似度合并（≥ merge_threshold 复用节点，否则新建），保证同一物理页面在不同任务下归并为同一节点，跨任务学习成立。

### 4.3 存储（AutoDL only）

| 端 | 角色 | 更新方式 |
| --- | --- | --- |
| **AutoDL 云端** | 唯一存储 + 向量检索 | `POST /add_transition` 增量：页面合并 + FAISS 追加 + 落盘 |

**架构**：本机 agent 在门控通过后调用 `record_transition` → **只**推送到 AutoDL。无本机 page-graph / 本地 embedding 回退；`rag_url` 缺失或远程失败时直接报错。

- 单 agent：步内缓冲，仅 `set_episode_success(True)` 时 flush。
- multiagent：仅 Action Verifier=`CORRECT` 的步立即写边。

### 4.4 检索

- `retrieve_hint`：构造当前屏幕摘要 → **仅**远程 AutoDL RAG → 注入 `## Environment Knowledge (U3)`。
- summary 为空时跳过远程（首步无屏幕信息）。
- 远程失败抛错，不回落到本地图。

### 4.5 消融开关

`--u3` 为唯一开关。启用时必须提供有效 `--rag_url` / `RAG_URL`。消融脚本里 `--norag_on` 为兼容别名，效果等同关闭 U3（不再存在 “local graph only”）。

---

## 5. U4 程序性技能记忆

### 5.1 定位

跨任务复用的可执行技能：参数化、可执行、可验证、可更新。从多条成功轨迹中抽象，是 U2 之上的压缩层。

### 5.2 数据结构

```
Skill:
  goal_hint    适用任务类别（检索键①，已抽象去具体参数）
  precondition 适用屏幕状态（检索键②，取技能起始屏幕）
  actions      参数化动作序列（语义 token，绝不存元素索引）
  slots        需填参数名列表
  score        历史表现，驱动保留/淘汰
  (元数据) successes / failures / version
```

```
SkillAction:
  action_type  动作类型（click/input_text/scroll/open_app...）
  target       语义选择器（text / content_description / hint_text）
  params       动作参数（可含 {slot} 占位）
  app          open_app 时的应用名
```

**关键约束**：target 用语义 token（元素文本/内容描述），不用索引——索引在重新渲染后失效，语义 token 跨任务稳定。

### 5.3 抽象流程（纯确定性，无 LLM）

```
多条成功轨迹（agent history，ground-truth 确认）
  → 语义 token 化（动作 + 元素语义）
  → 跨轨迹槽位抽象（同位置值不同 → {slot}）
  → BPE 式子序列挖掘（高频相邻动作对合并，EAM Action Group Mining）
  → goal_hint 抽象（去具体文件名/时间戳/参数 → 任务类别）
  → 存技能库
```

### 5.4 检索与更新

- **检索**：`goal_hint` + `precondition` 双因子，k=1（ReasoningBank 的 k=1 结论）。
- **更新**：执行成功 → `record_outcome(True)` 加分；失败 → score 递减，≤0 淘汰（ProcMEM 增益剪枝）。
- **数据独立性**：U4 从 agent 自己 history 取数，不依赖 U2 代码——`--u4` 单独开也能跑。

### 5.5 参考论文

Agent-Workflow-Memory（`{slot}` 参数化）、Mobile-Agent-E（precondition 门控）、SkillWeaver（可执行程序 + 测试）、ProcMEM（激活/执行/终止三元组 + 分数剪枝）、ReasoningBank（goal 通用化约束）、EAM（BPE 挖掘）。

### 5.6 消融开关

`--u4` 独立开关。

---

## 6. U5 内部化记忆

### 6.1 定位

**离线微调**。把"模型如何使用记忆"的能力内化进权重，而非显式规则。与 U1-U4（在线显式）根本区别在时态：U1-U4 运行时读写，U5 任务跑完后蒸馏。

### 6.2 与在线系统的关系

```
在线（U1-U4 + PER）              离线（U5）
  积累 (任务, 轨迹, 记忆使用, 成败) → 微调数据集 → LoRA 微调模型
```

U1-U4 + PER 跑完产生大量带标注轨迹（哪些记忆检索帮助了成功决策），U5 用这些数据把记忆使用能力学进模型权重。

### 6.3 当前状态

❌ 未实现。与多智能体阶段结合，留作后续工作。

---

## 7. 与多智能体的融合（PER v2.1）

### 7.1 融合原则

**Agents 读记忆，Verifier 写记忆。** 三合一 Verifier 是记忆的质量闸门——它把"什么算成功"从外部任务评价器细化到系统内部的逐项认证。

### 7.2 读路径（记忆 → Agents）

| Agent     | 读哪层                       | 时机   | 用途                                |
| --------- | ------------------------- | ---- | --------------------------------- |
| Planner   | U2 轨迹、U4 技能               | 开局分解 | 相似任务参考注入                          |
| Executor  | U1 任务态、U3 页面图、U2 轨迹、U4 技能 | 每步决策 | 屏幕指南 + 轨迹/技能提示                    |
| Reflector | 不读记忆                      | —    | 只产生信号 |

### 7.3 写路径（Verifier 信号 → 记忆）—— 融合枢纽

| Verifier 信号                                     | 喂给哪层        | 具体操作 |
| ----------------------------------------------- | ----------- | ---- |
| Action Verifier `CORRECT`                       | U3          | multiagent：该 UI 转换才写入 AutoDL 页面图 |
| Action Verifier `MISGROUNDED` / `NO_EFFECT`     | U3          | 不写入图；反馈 Executor（不单独加 stall） |
| Progress Auditor `ADVANCING`                    | U1          | 子目标级认证通过后更新台账 / 推进子目标 |
| Progress Auditor `STALLED` / `LOOPING` / `BUSY` | Planner     | stall 达阈值后 replan（受 `MAX_REPLANS`） |
| Evidence Certifier `PASS`                       | U2、U4       | 与外部 GT 融合后沉淀轨迹 / 技能 |
| Evidence Certifier `FAIL`                       | —           | 否决 `status/complete`；max-steps 可重认证 |

单 agent（无 multiagent）：无步级 AV；U3 在 episode GT 成功时整段 flush 到 AutoDL。U4 按 episode 成败 `record_outcome`，**不再**用 AV 做步级学分。

### 7.4 三合一 Verifier 对记忆的深层价值

1. **记忆只存可信的东西**（multiagent）：只有 Action Verifier 确认 CORRECT 的 UI 转换才写 U3。
2. **完成门控**：Evidence Certifier 否决过早的 `status/complete`；空 ACCEPTANCE 清单 fail-closed（不造假默认 A1）。
3. **Progress Auditor 触发 replan**：连续无进展达阈值后刷新子目标（不与 cert veto 共用 `MAX_REPLANS`）。

### 7.5 与现有代码的衔接

`set_episode_success`：外部 AndroidWorld GT 与 multiagent `_certified`（若已设定）AND 后写 U2/U4；单 agent 另将缓冲的 U3 转换 flush 到 AutoDL（失败不阻断 U2/U4）。

## 8. 当前进度

| 层        | 状态        | 说明 |
| -------- | --------- | ---- |
| U1 任务状态  | ✅ 完成      | task_state.py；multiagent 写入子目标 |
| U2 情景轨迹  | ✅ 数据层完成   | episodic.py 包装 DMS；replay finalize 已接 |
| U3 环境知识  | ✅ AutoDL-only | 仅远程图；无本机 page-graph 回退；缺 `rag_url` 报错 |
| U4 程序性技能 | ✅ 数据层完成   | 成功/失败均 `record_outcome`；技能以 prompt hint 注入 |
| U5 内部化   | ❌ 未实现     | 离线微调，留作后续工作 |

## 9. 参考论文

| 层   | 参考论文                                                                                                                              |
| --- | --------------------------------------------------------------------------------------------------------------------------------- |
| U2  | DMS（arXiv:2601.22528）                                                                                                             |
| U3  | PG-Agent（arXiv:2509.03536）                                                                                                        |
| U4  | Agent-Workflow-Memory、Executable-Agentic-Memory、Mobile-Agent-E、ProcMEM、ReasoningBank、SkillWeaver（`C:\Users\WRQ\Desktop\论文\记忆机制\`） |
