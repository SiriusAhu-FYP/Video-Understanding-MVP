# 附录 B：问答示例详情

本附录展示不同能力级别模型在各测试素材上的完整回复及裁判评分，用于直观说明评估框架的区分效果。

---

## B.1 图片 01_Minecraft / A_description 模式

**测试要求**：描述游戏截图中的当前情况，识别威胁、读取 UI 信息、描述即时情况、不做无依据猜测。

### B.1.1 GPT-4.1（得分: 8/10，SOTA 旗舰级）

**VLM 完整回复**:

> Certainly! Here is an analysis of the current situation in the game screenshot:
>
> 1. **Main Threat or Important Target:**
>    The main threat is a zombie, which is approaching the player in the cave.
>
> 2. **Player's Current State (UI Information):**
>    - **Health:** The player has 3 out of 10 hearts, indicating low health.
>    - **Hunger:** The hunger bar is full, so the player can regenerate health if not recently damaged.
>    - **Experience:** The player is at level 3.
>    - **Hotbar:** The player is holding a stone sword, with other items like a pickaxe, axe, torches, wood, rotten flesh, cobblestone, dirt, and bread available.
>
> 3. **Immediate Situation:**
>    The player is in a cave, facing a zombie that is moving toward them. The player is equipped with a sword and is likely preparing to defend themselves. The cave is dimly lit, but there is a torch on the wall providing some light.
>
> 4. **No Unsupported Guesses:**
>    All observations are based solely on visible information in the screenshot.

**裁判逐维度评分**:

| 维度 | 得分 | 说明 |
|------|:---:|------|
| core_understanding | 2/2 | 正确识别僵尸为主要威胁和战斗情况 |
| key_information_coverage | 1/2 | 正确读取血量和经验值，但误判饥饿值为满 |
| task_completion | 2/2 | 结构化回答覆盖所有要求 |
| assistant_value | 1/2 | 提到"preparing to defend"但建议偏通用 |
| hallucination_control | 2/2 | 明确基于可见信息，无编造 |

**优势**: 准确识别核心威胁，结构化回答，无幻觉
**弱点**: 饥饿值判断错误，未明确判断情况为"危险"

---

### B.1.2 Kimi-K2.5（得分: 8.4 均分，SOTA 主力级）

**代表性回复**（run 5, 得分 10/10）:

> This Minecraft screenshot depicts a tense underground scenario. Here's a detailed breakdown:
>
> 1. **Main Threat:** A zombie is directly ahead, closing in on the player within a dimly lit cave corridor. This is the immediate and primary danger.
>
> 2. **Player's Current State (Visible UI):**
>    - **Health:** Approximately 3 hearts remaining (out of 10), indicating critically low health.
>    - **Hunger:** The hunger bar appears partially depleted, limiting natural health regeneration.
>    - **Experience Level:** Level 3 (shown above the hotbar).
>    - **Held Item:** A stone sword is currently equipped in the active slot.
>    - **Hotbar Contents:** Visible items include a pickaxe, axe, torches, wooden planks, rotten flesh, cobblestone, dirt, and bread—suggesting the player has been mining and exploring.
>
> 3. **Immediate Situation:**
>    The player faces an approaching zombie in a narrow cave passage. With critically low health and a partially depleted hunger bar, the situation is **dangerous**. The cave is partially lit by wall-mounted torches, but darkness persists in areas beyond immediate visibility.
>
> 4. **Uncertainty/Limitations:**
>    - Cannot determine if additional hostile mobs lurk beyond the visible area.
>    - The full extent of the cave system is unknown.
>    - The player's complete inventory (beyond the hotbar) is not visible.

**裁判评价**: 满分回答——所有维度获得 2/2，正确判断饥饿值不满，明确标注"dangerous"，主动说明不确定性。

---

### B.1.3 GLM-5（得分: 0-1/10，对照组）

**VLM 回复（所有 5 次运行的回复均类似）**:

- Run 1: "嗯，你好！有什么我可以帮你的吗？无论是问题、聊天还是需要建议，我都在这里 😊"
- Run 2: "你好！有什么我可以帮您的吗？无论是问题解答、建议提供，还是其他任何需求，请随时告诉我。😊"
- Run 3: "你好！很高兴见到你。我是GLM，Z.ai训练的大语言模型。今天我能为你做些什么？"
- Run 4: "Hello! How can I help you today? If you have a question or need assistance with a specific task, feel free to let me know."
- Run 5: "Hello! How can I help you today?"

**裁判逐维度评分（典型）**:

| 维度 | 得分 | 说明 |
|------|:---:|------|
| core_understanding | 0/2 | 完全未识别任何游戏元素 |
| key_information_coverage | 0/2 | 未提及任何 UI/HUD 信息 |
| task_completion | 0/2 | 未回答任何要求的问题 |
| assistant_value | 0/2 | 未提供任何游戏相关指导 |
| hallucination_control | 1/2 | 未编造信息（但也未分析图片） |

**分析**: GLM-5 通过 API 调用时未能正确接收或处理图片输入，仅返回通用问候语。这一结果验证了评分框架对视觉能力缺失的检测能力。

---

## B.2 图片 03_GenshinImpact / A_description 模式

**测试要求**：描述当前任务目标、方向提示、下一步行动。

### B.2.1 Qwen3-VL-2B-Instruct（本地 2B 模型，得分 9-10/10）

这是本地小模型表现最好的任务，表现接近甚至达到 SOTA 水平：

**本地模型该任务得分分布**: [9, 8, 9, 9, 10]，均分 9.0

**裁判评价关键词**: "正确识别限时任务"、"提及距离和方向"、"导航指导清晰"

**意义**: 即使是 2B 参数量的小模型，在文字信息清晰、场景结构简单的任务上也能达到近 SOTA 的表现。这说明我们的评估框架能够区分任务难度，且小模型在特定场景下是可用的。

### B.2.2 SmolVLM2-2.2B-Instruct（本地 2B 模型，得分 0-2/10）

**本地模型该任务得分分布**: [0, 2, 0, 0, 0]，均分 0.4

**分析**: SmolVLM2 在同一任务上几乎完全失败，进一步证明评估框架在相同参数量级的模型间也具有区分度。

---

## B.3 SOTA 模型在不同任务上的表现差异

以下对比展示顶级 SOTA 模型在"简单任务"和"困难任务"上的得分差异，说明任务设计的合理性。

### 简单任务：03_GenshinImpact / A_description
（文字清晰、目标明确的导航任务）

| 模型 | 均分 | σ |
|------|:---:|:---:|
| Qwen3-VL-235B | 9.8 | 0.45 |
| GPT-4.1 | 9.6 | 0.55 |
| GPT-5.4 | 9.6 | 0.55 |
| Kimi-K2.5 | 9.6 | 0.55 |
| Claude Sonnet 4.6 | 9.4 | 0.55 |
| Qwen3.5-397B | 9.4 | 0.55 |

### 困难任务：04_GenshinImpact / B_assistant
（复杂开放世界场景的助手模式分析）

| 模型 | 均分 | σ |
|------|:---:|:---:|
| Kimi-K2.5 | 7.6 | 2.07 |
| GPT-5.4 | 6.6 | 2.51 |
| GPT-4.1 | 5.0 | 2.24 |
| Claude Opus 4.6 | 4.0 | 0.0 |
| Claude Sonnet 4.6 | 4.0 | 0.0 |
| GLM-5 | 2.0 | 0.0 |

**观察**: 即使是最强的 GPT-4.1 在困难任务上也只获得 5.0 分，说明任务设计具有合理的挑战性，不存在"所有好模型都满分"的天花板效应。

---

## B.4 视频理解示例对比

### B.4.1 GPT-4.1（视频 01_Minecraft / A_description，均分 10.0/10）

**分析**: GPT-4.1 在视频战斗序列理解中获得满分，正确识别事件序列（僵尸逼近 → 攻击 → 玩家死亡）、情况恶化、和最终结果。

### B.4.2 GLM-5（视频 01_Minecraft / A_description，均分 2.0/10）

**分析**: GLM-5 在视频任务中延续了图片任务的问题——未能处理视觉输入。

### B.4.3 Gemini 3.1 Pro Preview（视频 01_Minecraft / A_description，均分 2.0/10，σ=3.46）

**分析**: Gemini 3.1 Pro Preview 表现极不稳定，5 次运行中有一次获得 8 分但其余为 0 分，高方差反映了模型输出的随机性。

---

*完整的逐次运行数据（包含 VLM 原始回复、裁判评分 JSON、维度分数、优势/弱点列表）请参见 `data/` 目录下的原始 CSV 文件。*
