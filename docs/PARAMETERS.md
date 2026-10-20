# 参数说明

通过 `--set 参数名=值` 覆盖配置，例如 `--set seed=43`、`--set save_predictions=false`。

## Traffic 默认配置

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| data / features / enc_in | Traffic / M / 862 | 数据集、预测模式和变量数 |
| root_path / data_path | dataset/traffic / traffic.csv | 数据路径 |
| seq_len / pred_len | 96 / 96 | 输入和预测长度 |
| patch_len / stride | 16 / 8 | 分块长度和步长 |
| gpt_layers / hidden_size | 6 / 768 | GPT-2 层数和隐藏维度 |
| batch_size / eval_batch_size | 8 / 1 | 训练和评估批量大小 |
| epochs / patience | 50 / 10 | 最大训练轮数和早停耐心值 |
| learning_rate | 0.0005 | AdamW 学习率 |
| seed | 42 | 随机种子；批量运行使用 42/43/44 |
| device | cuda:0 | 运行设备，可改为 cpu |

## 常用参数

- `task`：long_term / low_data / m4。
- `percent`：使用最早训练窗口的百分比；低数据任务为 10/25/50。
- `bridge_layers`：桥接位置，默认 [1,3,5]，从 1 开始编号。
- `frequency`：M4 频率；`pred_len` 使用对应预测长度，`seq_len=2*pred_len`，`patch_len=4`、`stride=2`。
- `m4_refit`：按选定轮数在完整训练集上重训，默认 true。
- `naive2_path`：M4 Naive2 基准 CSV 路径。
- `backbone_path`：GPT-2 权重目录；`output_dir`：输出目录。
- `save_predictions`：是否保存预测数组，默认 true。
- `num_workers`：数据加载进程数，默认 0；`deterministic`：是否启用确定性算法，默认 false。

完整默认参数见 `configs/base.json`，每次运行的实际参数保存为 `config.json`。

默认训练范围固定：GPT-2 词嵌入、Attention 和 FFN 冻结；位置嵌入、LayerNorm 及新增模块参与训练。

当前 Linux 默认：基础 batch_size=512，ETT/M4 继承；Weather=16，Traffic/Electricity=8。默认 epochs=50、patience=10。硬件和启动方式见 [README](../README.md)。
