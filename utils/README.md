# Utils — 实验共享工具

本目录存放实验脚本间共享的工具模块，不属于 `ahu_paimon_toolkit` 核心包。

## 模块说明

| 模块 | 用途 | 主要导出 |
|------|------|---------|
| `csv_io.py` | CSV 文件读写 | `init_csv()`, `append_csv()`, `read_csv_dicts()` |
| `logging.py` | loguru 文件日志设置 | `setup_experiment_log()` |
| `reporting.py` | 模板驱动报告生成 + DeepSeek 分析 | `fill_template()`, `generate_report_from_template()`, `write_report()` |
| `vllm_manager.py` | WSL vLLM 服务生命周期管理 | `start_vllm()`, `stop_vllm()`, `get_vllm_base_url()` |

## 使用示例

```python
from utils.csv_io import init_csv, append_csv
from utils.logging import setup_experiment_log
from utils.reporting import generate_report_from_template
from utils.vllm_manager import start_vllm, stop_vllm
```

## 报告生成流程

`utils/reporting.py` 实现模板驱动的报告生成：

1. 从 `templates/` 加载 Markdown 模板
2. 用 `{{placeholder}}` 语法替换数据（表格、统计信息）
3. 将填充后的模板发送给 DeepSeek，对 `<!-- ANALYSIS -->` 标记的部分生成深入分析
4. 写入最终报告文件

```python
generate_report_from_template(
    "speed_comparison.md",           # 模板文件名
    {"speed_table": table_text},     # 数据字典
    output_path / "report.md",       # 输出路径
    use_deepseek=True,               # 是否使用 DeepSeek 分析
)
```
