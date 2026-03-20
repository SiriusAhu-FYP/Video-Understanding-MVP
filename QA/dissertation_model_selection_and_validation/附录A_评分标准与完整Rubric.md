# 附录 A：评分标准与完整 Rubric

本附录包含实验中使用的所有评分标准的完整定义。每个测试素材均包含独立的 rubric，但共享统一的五维度评分框架。

---

## A.1 统一评分框架

所有素材使用 **5 维度 × 0-2 分** 的评分框架，满分 10 分。

| 维度 | 英文标识 | 含义 | 评分标准 |
|------|---------|------|---------|
| 核心理解 | core_understanding | 模型是否正确识别核心情况和主要威胁/目标 | 2=清晰准确, 1=部分正确, 0=错误或遗漏 |
| 关键信息覆盖 | key_information_coverage | 模型是否覆盖重要的可见信息（UI/HUD等） | 2=覆盖大部分, 1=覆盖部分, 0=遗漏大部分 |
| 任务完成度 | task_completion | 回答是否完成了 prompt 要求的所有部分 | 2=完整回答, 1=部分回答, 0=未能有效回答 |
| 助手实用价值 | assistant_value | 回答是否提供有用、可操作的游戏助手指导 | 2=清晰实用, 1=有用但泛泛, 0=缺失或无关 |
| 幻觉控制 | hallucination_control | 模型是否避免无依据推测并正确处理不确定性 | 2=严格基于事实, 1=基本可靠, 0=编造信息 |

---

## A.2 图片素材 01_Minecraft — 战斗 HUD 读取 + 战术建议

### 场景描述
玩家在洞穴中面对一只逼近的僵尸。可见 HUD 显示血量、饥饿值、经验等级、手持物品和快捷栏。

### A_description Prompt
> Please describe the current situation in this game screenshot. Your answer should: (1) identify the main threat or important target, (2) summarize the player's current state using visible UI information, (3) describe the immediate situation shown on screen, and (4) avoid guessing anything that is not supported by the image.

### B_assistant Prompt
> You are a game assistant. Based only on this screenshot: (1) what is the most important thing the player should pay attention to right now? (2) is the current situation safe, neutral, or dangerous? (3) what visible evidence supports your judgment? (4) what cannot be determined from this screenshot alone?

### 参考答案要点
- 玩家在洞穴中
- 正前方的僵尸是主要威胁
- 玩家手持剑
- 血量低
- 饥饿值不满
- 经验等级为 3
- 快捷栏中有工具、方块和食物/资源
- 当前情况危险

### 维度评分标准

**core_understanding**:
- 2: 清晰识别僵尸为主要威胁，认识到战斗情况
- 1: 提到了危险或敌人但模糊不完整
- 0: 遗漏威胁或误判情况

**key_information_coverage**:
- 2: 正确提到大部分关键 UI 要点，包括低血量和其他可见状态/资源信息
- 1: 提到部分 UI 信息但遗漏重要部分
- 0: 遗漏大部分关键 UI 信息或读取错误

**task_completion**:
- 2: 直接且完整地回答了所有要求的问题
- 1: 部分回答了任务
- 0: 未能以有用的方式回答任务

**assistant_value**:
- 2: 基于截图提供清晰、相关、可操作的建议
- 1: 建议有一定用处但较为通用
- 0: 建议缺失、无关或无帮助

**hallucination_control**:
- 2: 避免猜测，明确区分可见事实和不确定性
- 1: 基本可靠但未明确说明不确定性
- 0: 做出无依据的声明或明显的幻觉

### 严格扣分项
- 说当前情况安全
- 未能识别僵尸为主要威胁
- 忽视玩家低血量
- 编造图像中不存在的敌人、物品或事件

---

## A.3 图片素材 03_GenshinImpact — 任务文本 / 导航指引理解

### 场景描述
原神任务场景，玩家需要在限时内到达目的地。左侧可见任务文本，包括方向提示（目标在下方 44 米处）。

### A_description Prompt
> Please describe the current objective in this game screenshot. Your answer should: (1) explain the quest goal, (2) summarize the visible directional hints, (3) describe what the player is expected to do next, and (4) avoid guessing anything not supported by the image.

### B_assistant Prompt
> You are a game assistant. Based only on this screenshot: (1) what should the player do next? (2) what visible evidence supports your judgment? (3) is the key challenge here combat or navigation? (4) what cannot be determined from this screenshot alone?

### 参考答案要点
- 当前任务是在限时内到达目的地
- 任务进度 0/1
- 可见提示显示目标在下方
- 目标距离约 44 米
- 玩家应沿可见路径向前、向下移动
- 这是导航任务，非战斗任务

### 维度评分标准

**core_understanding**:
- 2: 清晰识别限时到达目的地任务和导航性质
- 1: 部分识别目标但遗漏限时或导航方面
- 0: 误判任务或未能正确描述目标

**key_information_coverage**:
- 2: 正确提到任务目标、0/1 进度、下方方向和大致距离
- 1: 提到部分重要信息
- 0: 遗漏大部分关键信息或读取错误

**task_completion**:
- 2: 使用可见证据清晰解释下一步行动
- 1: 给出部分正确的下一步但缺乏证据或清晰度
- 0: 未能以有用方式推断下一步

**assistant_value**:
- 2: 提供清晰、可操作的导航指导
- 1: 提供有一定用处但通用的指导
- 0: 提供很少或没有有用的帮助

**hallucination_control**:
- 2: 避免无依据猜测并明确区分不确定性
- 1: 基本可靠但未明确说明不确定性
- 0: 编造倒计时、敌人或路线细节

### 严格扣分项
- 声称场景主要关于战斗
- 忽视"下方"方向提示
- 编造不存在的倒计时器
- 对敌人或路线做出无依据的强烈声明

---

## A.4 视频素材 01_Minecraft — 战斗序列理解

### 场景描述
视频片段展示玩家在洞穴中遭遇僵尸攻击。玩家情况随时间恶化，最终死亡。

### 维度评分标准（视频专用维度名称）

**event_sequence_understanding**:
- 2: 清晰描述僵尸遭遇、危险升级和最终死亡的正确序列
- 1: 部分描述了序列但遗漏或混淆某些步骤
- 0: 未能理解序列或描述错误

**key_change_recognition**:
- 2: 清晰表明情况恶化、危险升级
- 1: 模糊提到变化但未清晰解释
- 0: 未能识别重要变化

**outcome_understanding**:
- 2: 正确表明玩家死亡并归因于僵尸
- 1: 部分正确但不完整或模糊
- 0: 误判或遗漏最终结果

**assistant_value**: 同图片标准

**hallucination_control**: 同图片标准

### 严格扣分项
- 未能识别僵尸为主要威胁
- 遗漏玩家最终死亡
- 声称不同的死因（无依据）
- 编造视频中未发生的重大事件

---

*其余素材（02_Minecraft, 04_GenshinImpact, 02_GenshinImpact_video）的完整 rubric 结构与上述一致，详见原始 JSON 文件：`assets/images/*/\*.json` 和 `assets/videos/*/\*.json`。*
