# U2 插槽回放(Slot-Based Replay)设计文档

> 日期:2026-08-16
> 状态:v0.1 草稿
> 定位:U2 回放路径的增强,解决"遇参数停导致回放复用打折"。

---

## 1. 背景与问题

### 1.1 现状

U2 在单智能体下按**整个任务**粒度存取轨迹(任务粒度,不拆子目标):

- **存**:任务结束时,整条轨迹按 `Plan(precondition=起始屏幕, goal=完整任务)` 存进 DMS bank。
- **取**:**hint**(每步注入动作类型串) + **replay**(开局命中就整条回放)。
- **回放**([memory_agent.py](androidworld/android_world/agents/memory_agent.py) `_step_replay`):导航动作直接执行;遇到 `input_text` **终止回放,交回 LLM**(当前行为,建议一)。

### 1.2 问题

多参数、跨 APP 任务(如"在 Markor 写笔记,通过短信发给联系人")里,轨迹包含多个 `input_text`。当前"遇第一个 input_text 就整体终止"导致:

- 回放只覆盖第一个参数之前的一小段导航;
- 之后的导航动作(点分享、选短信应用等)全部丢给 LLM 重做;
- **参数越多,回放复用得越少**。

这不是"参数安全"问题(参数安全已被建议一解决),而是**复用效率**问题。

### 1.3 目标

让回放在多参数任务里**持续复用导航动作**,只有参数位花 LLM 现填:

- 导航动作(`click`/`scroll`/`open_app`/`navigate_*`/`keyboard_enter`)→ 回放直接执行;
- 参数位(`input_text`)→ 停下,LLM 对着当前屏幕 + 任务目标**现填**,填完**继续回放**。

## 2. 设计约束(必须守住)

| # | 约束 | 理由 |
|---|---|---|
| C1 | 填槽代码**只写在 `_step_replay` 内**,直接调 `self.llm`,**不跳回 `super().step()`** | baseline 从不走回放路径,零影响;代码边界干净 |
| C2 | **U2 数据层([episodic.py](androidworld/android_world/agents/memory/episodic.py))不改、不调 LLM** | 保住"U2 是纯记忆层"的设计声明 |
| C3 | 存储格式不变(仍存原始轨迹,含旧参数值) | 旧值只在回放时被"现填"覆盖,不直接执行 |
| C4 | 建议二(precondition 起始屏幕)保持 | 存取的 `Plan(pre, goal)` key 继续对齐 |
| C5 | 单智能体**不接子目标分解** | `_decompose_into_subplans` 保持占位,任务粒度不变 |
| C6 | 填槽用**纯文本** prompt(不截图) | 省 token,填字场景够用 |

## 3. 机制

### 3.1 回放时动作分流

在 `_step_replay` 中,按 `action.action_type` 分流:

| 动作类型 | 处理 | 执行者 |
|---|---|---|
| `click` / `double_tap` / `long_press` / `scroll` / `open_app` / `navigate_home` / `navigate_back` / `keyboard_enter` / `wait` | 直接执行(click 索引照旧重绑) | 回放 |
| `input_text` | **停下 → LLM 现填 → 继续回放** | 槽,LLM |
| `status`(complete / infeasible) | 结束回放,交回 | — |

### 3.2 填槽的 LLM 调用

每个 `input_text` 停下时,调一次 LLM(纯文本):

```
当前屏幕 UI 元素文本(纯文本,来自 env.get_state 的 ui_elements_list)
任务目标(原始 goal)
槽位说明:"这是回放轨迹第 N 步,需要填入一个文本值"
```

LLM 输出一个 `input_text` 值 → 转成 `JSONAction(action_type="input_text", text=..., index=槽目标 index)` → 执行 → 回放继续到下一个动作。

- **index 处理**:复用现有 `_resolve_action_target` 语义,把槽目标的文本/内容描述绑定到当前屏幕元素(与 click 重绑同机制)。
- **失败处理**:LLM 输出非法/空 → 终止回放,交回主循环(不阻塞、不重试循环)。
- **token 统计**:每次填槽的 LLM 调用计入 episode token 用量。

## 4. 消融配置

U2 不拆测(用户决定):只保留完整形态(含插槽回放)。8 个配置,跨 seed 独立 store。

### 主实验 2×2(4 个)

| 配置 | 说明 |
|---|---|
| Baseline | 无记忆,无多智能体 |
| +MA | 多智能体(AV/PA/EC) |
| +Mem | 记忆完整形态(U2 含插槽回放) |
| +MA+Mem | 全开 |

### 记忆消融·单智能体(4 个)

| 配置 | 说明 |
|---|---|
| u1 | 任务状态记忆 |
| u12 | +U2 插槽回放 |
| u123 | +U3(AutoDL RAG) |
| u1234 | +U4 程序性技能 |

## 5. 指标

| 类别 | 指标 |
|---|---|
| 主指标 | 任务成功率(跨 seed mean±std) |
| 辅指标 | 每任务步数、token 用量、完成时间 |
| 机制指标 | U2 检索命中率、replay 触发次数、填槽次数、填槽成功/失败率、导航复用步数 |

## 6. 实现范围

**改动文件**:
- `android_world/agents/memory_agent.py`:`_step_replay` 内加 `input_text` 填槽分支 + 填槽 prompt 构建 + 填槽 LLM 调用。
- (可选)`android_world/agents/memory/` 新增填槽 prompt 模板常量或小工具,但 **episodic.py 不改**。

**测试**:
- 单元测试:填槽 LLM 输出合法 `input_text`;非法输出降级交回主循环。
- 集成测试:多参数轨迹回放流程(导航执行 → 填槽 → 继续导航)。

**不改**:
- `episodic.py`(U2 数据层)、`_decompose_into_subplans`(保持任务粒度)、`multi_agent.py`。

## 7. 相关现状盘点(设计时发现)

| 位置 | 状态 |
|---|---|
| U4 `validate_skills()` | 文档([procedural.py:12](androidworld/android_world/agents/memory/procedural.py:12))承诺,方法不存在(技能挖出未验证) |
| `page_graph.py` | 完整实现但实际 U3 走 AutoDL,本地图从未被用 |
| `.tmp_app.py` | 引用的 embedder.py/retrieve.py 不存在,孤立残缺文件 |
| AV 确定性快速路径 | 文档 §5.3 承诺,代码明确"Deterministic fast-paths are not implemented" |

以上不在本次实现范围内,仅记录。
