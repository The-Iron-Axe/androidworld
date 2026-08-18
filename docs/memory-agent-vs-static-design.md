# 记忆保管者(独立 Agent)vs 静态注入:实验设计论证

日期:2026-08-18
状态:设计论证(不写代码)

## 1. 问题

回忆一下现状与目标:

- 当前架构里,记忆(U1-U4,`memory_agent.py`)是**纯数据基础设施**,直接拼进主 agent 的 action prompt 作为静态上下文块。**读写记忆几乎不额外调 LLM**(唯一例外是 U2 确定性回放的 slot-fill 填槽,见 `memory_agent.py:659`)。
- 很多论文/系统(MemGPT、AgentStore、带"记忆保管者"的多智能体)把记忆做成**一个带独立对话循环、独立 prompt 的 agent 节点**——其他 agent 通过消息请求读写,保管者自己调 LLM 检索/摘要/决策后返回加工结果。

**要回答的问题**:这两种形态,哪个更有效 / 更省 token?

## 2. 两种形态的对比

| | 静态注入(现状) | 记忆保管者(独立 Agent) |
|---|---|---|
| 记忆形态 | U1-U4 纯数据结构,注入 action prompt | 独立对话循环、独立 prompt 的 agent 节点 |
| 读写路径 | 读对象 / 查本地库 / U2 回放 / U3 RAG,**零额外 LLM 调用**(除 slot-fill) | 其他 agent 发消息 → 保管者自己调 LLM → 返回加工结果 |
| 边际成本 | ≈ 0(注入的几段静态文本) | 每次记忆访问 +1 次 LLM 调用 + 上下文 token |
| 对应文献 | 记忆 = 检索增强上下文 | 记忆 = 一个自主角色 agent |

**本质差异**:是"记忆内容免费呈现在上下文里",还是"每次访问都要一次 LLM 加工"。

## 3. 可测假设

- **H1(有效)**:记忆保管者化后,任务成功率 ≥ 静态注入。
- **H2(成本)**:记忆保管者化的 token 与调用次数 > 静态注入(几乎必然,因为每访问 +1 调用)。
- **结论形态**:一张 **有效 vs token 的 trade-off**,让读者判断"A 点成功率换取 B 点 token 是否值得"。

## 4. 方法论陷阱:算力 confounder(命门)

**这是整个实验最关键的地方。**

- 静态注入 = 记忆内容免费给你看,零额外调用。
- 记忆保管者 = 同样的记忆内容,但每次访问经过一次 LLM 加工,多花 1 次调用 + 额外 token。

如果保管者成功率更高,你怎么知道是"记忆 agent 化更有效",而不是"多花了一次 LLM 的思考"?这是**架构结论 vs 算力结论**的区别,不是同一件事。

**控制方案(可选的 arm B,严格化)**:给静态注入也配一条"每决策多一次 LLM 调用"的对照——把当前免费的上下文变成一次显式的中间推理调用,把两条 arm 的 per-step LLM 预算对齐。这才把 confounder 剥离:

- **A vs C** = 真实世界差异(含算力)。
- **B vs C** = 剥离算力后的因果差异。

如果 B 与 C 成功率相当但 C 能省 token(少走弯路),结论就硬了。

## 5. 建议的实验臂(推荐方案)

**主对比(两臂)**:
| 臂 | 记忆形态 | 记忆访问成本 | 启动命令 |
|---|---|---|---|
| A 静态注入(现状) | U1-U4 直接注入 action prompt | ≈ 0(除 U2 slot-fill) | `--multiagent --configs=u1234` |
| C 记忆保管者 | MemoryNode 独立节点,每步一次 LLM 加工 | 每次访问 +1 次 LLM(`mem` 模块) | `--multiagent --configs=u1234 --mem_as_agent` |

**MemoryNode 已实现**(2026-08-18):
- `memory_agent.py::MemoryNode` — 独立 prompt `MEMORY_NODE_PROMPT`,经 `begin_module('mem')` 计费,失败 fail-safe 返回空串。
- `MemoryAugmentedAgent.mem_as_agent` 开关:开时 `_build_action_prompt` 把 U1-U4 原始上下文交给 MemoryNode 蒸馏成单段 `## Memory (node)`;关时走静态注入(现状不变)。
- 开关透传:ablation `--mem_as_agent`、run.py `--mem_as_agent`,经 `**kwargs` 一路传给 MemoryAugmentedAgent。
- 单元测试:`multi_agent_test.py::MemoryNodeTest`(3 例),全套 51 例通过。

**B 等成本对照(可选,审稿强化)**:在 A 上加等成本额外决策调用,预算对齐 A/C。**先不做**。

**为什么先不做 B**:主问题由 A vs C 直接回答;若差异本就不显著,B 纯属浪费;若显著,B 再补。避免一开始就背上多一条对照臂的 token 开销。

## 6. 指标

### 有效性
- 每任务成功率(已由现有 pipeline 记录)。

### 成本(利用刚完成的 per-module 调用计数)
- 每任务 `num_calls`(全局 + per-module)。
- 每任务 `token_usage`(`prompt` / `completion` / `cache_hit`)。
- **记忆保管者计为独立模块 `mem`**,与 planner/av/pa/ec/main 分开计费。

### 新增维度(本次设计确认要加)
每 episode 额外记录:
- 记忆**读次数** / **写次数**;
- 其中**走了 LLM 的次数**(静态注入下 = 仅 slot-fill;保管者下 = 每次访问)。
- 记忆模块的 token(`mem` 模块的 prompt/completion)。

这一维度直接量化"记忆这层到底花了多少 token",是 trade-off 图的横轴燃料。

### trade-off 可视化
- x = 每任务平均 token(或调用次数),y = 成功率。
- A、C 两点(+ 可选 B)+ 误差棒。
- 让读者看到:记忆保管者比静态注入贵多少、换来多少成功率。

## 7. 公平性约束

- 同一批 task 集合、同一 seed、同一 LLM 后端。
- 除记忆形态外,其余多智能体模块(Planner/AV/PA/EC)保持不动。
- U1-U4 的记忆**内容**在两条 arm 下一致——差异只在"由保管者 LLM 加工后再给" vs "直接注入"。这样才干净地测"包装成 agent"这个动作本身。

## 8. 为什么现在时机合适

刚完成的 **per-module `num_calls` 计数**(`infer.py` `_module_calls` + `begin_module`/`end_module`,按模块计费)正是这个实验的地基:记忆保管者作为 `mem` 独立模块计数,才能精确算出记忆层成本。若无此计数,保真对比测不准。

## 9. 明确不做的范围

- **不做 arm B**(等成本对照)——仅作为可选强化保留;用户已确认忽略。
- **不做多轮 MemoryNode 对话循环/消息接口**——当前实现是"每步一次 node 加工 LLM"的最小可测形态,足以支撑 arm A vs arm C 主对比。

## 10. 落地后待办

1. **冒烟实验**:同一 `--multiagent --configs=u1234` 跑 1 seed,关/开 `--mem_as_agent` 各一遍;确认 arm C 每步比 arm A 多出 `mem` 模块调用(计数正确),成功率可对比。
2. 全量跑 A vs C(定 seed 数),产 trade-off 图。
3. 可选:把"记忆成本明细"写进 ablation 结果 JSON(读/写次数、其中走 LLM 次数)。
