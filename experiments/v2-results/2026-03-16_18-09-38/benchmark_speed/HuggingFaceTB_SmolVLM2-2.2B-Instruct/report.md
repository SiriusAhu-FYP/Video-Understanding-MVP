# VLM 基准测试报告: HuggingFaceTB/SmolVLM2-2.2B-Instruct

**生成时间**: 2026-03-16 20:24:56
**GPU 显存**: 15076 MB (运行时)
**总推理次数**: 90

## 各场景性能汇总

| 场景 | 平均 TTFT (s) | 平均吞吐量 (tok/s) | 平均总耗时 (s) | 平均输出 Tokens | 测试次数 |
|------|---------------|--------------------:|---------------:|----------------:|---------:|
| 短文本描述 | 0.119 | 117.2 | 0.515 | 46 | 18 |
| 详细描述 | 0.109 | 109.6 | 3.510 | 375 | 18 |
| JSON 物体检测 | 0.116 | 117.2 | 4.034 | 454 | 18 |
| 角色动作分析 | 0.113 | 110.5 | 1.399 | 145 | 18 |
| 游戏 UI 识别 | 0.107 | 120.3 | 1.888 | 210 | 18 |

## 各图片性能汇总

| 图片 | 平均 TTFT (s) | 平均吞吐量 (tok/s) | 平均总耗时 (s) | 测试次数 |
|------|---------------|--------------------:|---------------:|---------:|
| Arknights_01.png | 0.133 | 119.9 | 2.859 | 15 |
| Cyberpunk-2077_01.png | 0.114 | 115.3 | 2.596 | 15 |
| Genshin-Impact_01.png | 0.098 | 119.6 | 1.465 | 15 |
| Minecraft_01.png | 0.139 | 112.8 | 2.386 | 15 |
| P5R_01.png | 0.100 | 109.6 | 1.672 | 15 |
| SuperMarioBros_01.png | 0.093 | 112.5 | 2.635 | 15 |

## 全局统计

- **平均 TTFT**: 0.113s (std: 0.023s)
- **平均吞吐量**: 114.9 tokens/s (std: 12.7)
- **平均总耗时**: 2.269s
- **平均输出 Tokens**: 246
- **总推理次数**: 90

## 输出示例（每场景首次运行）

### 短文本描述 (`short_desc`)

**图片**: Arknights_01.png | **Tokens**: 64 | **耗时**: 0.6544s

**Prompt**: 用一句话简要概括这张游戏截图的画面内容。...

**输出**:
```
 The image depicts a screenshot of a mobile game, likely a strategy or battle royale game, featuring a grid-based battlefield with various characters and items. The game interface includes a scoreboard at the top, showing the player's current score and the number of enemies they have eliminated. The game is set in a
```

### 详细描述 (`detailed_desc`)

**图片**: Arknights_01.png | **Tokens**: 512 | **耗时**: 4.7014s

**Prompt**: 请用中文详细描述这张游戏截图中的所有元素，包括角色、环境、颜色、光照、构图、以及任何可以观察到的细节。尽可能全面。...

**输出**:
```
 这张游戏截图展示了一个游戏玩家在一个游戏场地上的游戏，玩家的角色在游戏场地上，玩家的角色有多个，玩家的角色有不同的装备和技能。玩家的角色在游戏场地上有不同的位置，玩家的角色有不同的功能，玩家的角色有不同的效果。玩家的角色在游戏场地上有不同的效果，玩家的角色在游戏场地上有不同的效果。玩家的角色在游戏场地上有不同的效果，玩家的角色在游戏场地上有不同的效果。玩家的角色在游戏场地上有不同的效果，玩家的角色在游戏场地上有不同的效果。玩家的角色在游戏场地上有不同的效果，玩家的角色在游戏场地上有不同的效果。玩家的角色在游戏场地上有不同的效果，玩家的角色在游戏场地上有不同的效果。玩家的角色在游戏场地上有不同的效果，玩家的角
```

### JSON 物体检测 (`object_detection`)

**图片**: Arknights_01.png | **Tokens**: 512 | **耗时**: 4.4654s

**Prompt**: 识别这张游戏截图中的所有主要物体和角色，输出 JSON 格式，每个对象包含 bbox_2d (左上角x, 左上角y, 右下角x, 右下角y) 和 label。格式示例：{"objects": [{"b...

**输出**:
```
 {
  "objects": [
    {
      "bbox_2d": [10, 10, 100, 100],
      "label": "Enemy"
    },
    {
      "bbox_2d": [100, 10, 100, 100],
      "label": "Enemy"
    },
    {
      "bbox_2d": [10, 100, 100, 100],
      "label": "Enemy"
    },
    {
      "bbox_2d": [100, 100, 100, 100],
      "label": "Enemy"
    },
    {
      "bbox_2d": [10, 100, 100, 100],
      "label": "Enemy"
    },
    {
      "bbox_2d": [100, 100, 100, 100],
      "label": "Enemy"
    },
    {
      "bbox_2d": [10, 100, 100,
```

### 角色动作分析 (`action_analysis`)

**图片**: Arknights_01.png | **Tokens**: 256 | **耗时**: 2.4415s

**Prompt**: 分析这张游戏截图中角色正在执行什么动作或行为。描述角色的姿态、朝向、与环境的交互，以及可能正在进行的游戏操作。...

**输出**:
```
 The image depicts a scene from a mobile game, likely a strategy or battle royale game, given the presence of multiple characters and the grid-based layout. The characters are arranged in a grid-like formation, with some characters positioned on the left side of the grid and others on the right. The grid is divided into squares, with each character occupying a specific square.

The characters are depicted in various poses and are equipped with different weapons and abilities. Some characters are
```

### 游戏 UI 识别 (`ui_recognition`)

**图片**: Arknights_01.png | **Tokens**: 256 | **耗时**: 1.975s

**Prompt**: 识别这张游戏截图中所有可见的 UI/HUD 元素，例如：血条、蓝条、小地图、技能图标、物品栏、计分板、对话框等。逐一列出每个 UI 元素的位置（画面的哪个区域）和当前显示的状态/数值。...

**输出**:
```
 1. 血条: 100%
2. 蓝条: 100%
3. 小地图: 100%
4. 技能图标: 100%
5. 物品栏: 100%
6. 计分板: 100%
7. 计数: 100%
8. 计数: 100%
9. 计数: 100%
10. 计数: 100%
11. 计数: 100%
12. 计数: 100%
13. 计数: 100%
14. 计数: 100%
15. 计数: 100%
16. 计数: 100%
17. 计数: 100%
18. 计数: 100%

```
