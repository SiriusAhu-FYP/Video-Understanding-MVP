# Video-Understanding-MVP 项目上下文与架构说明

## 1. 我们在做什么？(Project Overview)
本项目是一个非交互式的纯后台流水线脚本。

**核心逻辑：** 在 Windows 端播放一段视频（如 15 秒的猫狗追逐），程序在后台自动高频截图，通过计算前后帧的差异（状态增量）提取出“关键帧”。这些关键帧被放入异步队列，发送给本地部署的 vLLM 获取局部动态描述，最后将所有描述打包发送给云端 DeepSeek API，生成对整个视频的总结。

## 2. 运行环境与依赖 (Environment)
* **代码执行端:** Windows 11 (原生 PowerShell)。所有 Python 业务逻辑都在这里运行。
* **依赖管理:** `uv`。
* **本地 AI 节点 (已就绪):** vLLM 运行在 WSL 中，暴露的 OpenAI 兼容接口地址为 `http://localhost:8000/v1`。
    * 当前挂载模型：`Qwen/Qwen3-VL-2B-Instruct`。
    * 引擎端已配置了 `--enable-prefix-caching` 和显存分辨率限制，无需在 Windows 客户端代码中处理这些底层逻辑。
* **云端 AI 节点:** DeepSeek API。(通过`.env`中的`DEEPSEEK_API_KEY`和`DEEPSEEK_API_BASE_URL`配置)

## 3. 全局配置要求 (Config Driven)
程序必须是配置驱动的。所有的魔法数字（Magic Numbers）必须统一抽离到项目根目录的 `config.toml` 文件中。代码在启动时读取该配置。
需要写在 `config.toml` 里的参数包括但不限于：
* `[capture]`：目标窗口类名（如 `PotPlayer`）（这个参数应该需要完全匹配，而是在列表中模糊匹配最佳符合项）、基础截图频率（如 `screen_shot_interval_ms = 500`）、图片压缩长边上限（如 `max_size = 512` 会让如 1920x1080 的图片压缩到 512x288）、视频文件和播放器exe路径（可选，如果二者都提供了则默认在开启`main.py`时自动播放该视频）。
* `[algorithm]`：帧差异对比的算法选项（如 MSE/SSIM）、判定为“关键帧”的阈值（`diff_threshold`）。
* `[queue]`：异步队列的最大长度（防内存溢出）、关键帧的过期时间（`expiry_time_ms`，过期的旧图直接丢弃不送入推理）。
* `[llm]`：本地 vLLM 的 base_url、DeepSeek 的 API_KEY 和 base_url。
   * 补充：本地 vLLM 默认由我手动开启，端口在 `localhost:8000`。详情请搜索查阅 vLLM 的相关文档。
   * 补充：云端 DeepSeek 的 API_KEY 和 base_url 在 `.env` 中配置。
 * `[log]`: 日志输出路径（如 `log_file = "logs/xx-xx-xx_xx-xx-xx/xxx.log"`, 可以同时输出到控制台和文件）。

## 4. 核心工作流与技术规范 (Core Pipeline)
在后续编写代码时，必须严格遵循以下阶段和原则：

1. **画面捕获与差异对比 (Sync/Fast):**
   * 使用 `win32gui` 获取窗口，`mss` 极速截图。
   * 立即用 OpenCV 计算当前帧与上一关键帧的差异。超过 `diff_threshold` 才保留并转为 Base64，否则直接丢弃。
2. **异步队列与淘汰机制 (Async/Queue):**
   * 必须使用 `asyncio` 配合队列管理图片。如果 vLLM 处理不过来导致队列堆积，超时的图片必须被剔除，保证送入模型的是“最新且有效”的画面。
3. **vLLM 本地推理 (Async/VLM):**
   * 使用固定写死的 Prompt（如提取对象的 bounding box 和动作），调用 `localhost:8000`。
   * 此步骤完全异步，绝对不能阻塞 Windows 端的截图线程。
4. **DeepSeek 汇总 (Sync/LLM):**
   * 收集完一段时间内（或队列处理完毕后）的时序 JSON 数组，调用 DeepSeek API 进行最终的长文本归纳。

## 5. 关于提交
- 总是以 `Conventional Commits` 的规范提交代码。
- 总是在实现特定功能后进行一次提交，而不是囤积多个功能后再一次性提交。（注意：提交信息应当描述每个功能的实现细节）
- 总是为每个功能添加相应的测试用例。

## 其他
1. 总是使用 `from loguru import logger as lg` 作为日志记录器。
2. 总是及时更新README.md以反映项目的最新状态。（可能会有功能增删）