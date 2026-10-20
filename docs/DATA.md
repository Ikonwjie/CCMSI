# 数据准备

## 下载与目录

- ETT：[ETDataset](https://github.com/zhouhaoyi/ETDataset)
- Weather、Electricity、Traffic：[Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- M4：[M4-methods](https://github.com/Mcompetitions/M4-methods)

在项目根目录执行：

```bash
python scripts/prepare_data.py --dataset ETT
python scripts/prepare_data.py --dataset Weather
python scripts/prepare_data.py --dataset Electricity
python scripts/prepare_data.py --dataset Traffic
python scripts/prepare_data.py --dataset M4
```

ETT 命令一次下载四个子集。Weather、Electricity、Traffic 从 Time-Series-Library 的 [Hugging Face 数据仓库](https://huggingface.co/datasets/thuml/Time-Series-Library) 下载对应 CSV。每次下载会解析源仓库的具体提交版本，并在目标目录的 `download_manifest.json` 中记录该版本、下载地址和文件 SHA-256。可用 `--revision` 指定源仓库版本，用 `--output` 指定输出目录；已有文件不会被覆盖。

数据放置如下，M4 的 Naive2 压缩包需解压为 CSV：

```text
dataset/
  ETT-small/{ETTh1,ETTh2,ETTm1,ETTm2}.csv
  weather/weather.csv
  electricity/electricity.csv
  traffic/traffic.csv
  m4/
    Train/{Yearly,Quarterly,Monthly,Weekly,Daily,Hourly}-train.csv
    Test/{Yearly,Quarterly,Monthly,Weekly,Daily,Hourly}-test.csv
    submission-Naive2.csv
```

普通 CSV 保留 `date` 列和数值列；M4 第一列为序列 ID。

## 划分与预处理

- ETT 按时间顺序划分为训练 12 个月、验证 4 个月、测试 4 个月；Weather、Electricity、Traffic 按 7:1:2 划分。
- 标准化参数仅由训练数据计算，验证和测试复用；低数据任务使用指定比例的最早训练窗口。
- M4 使用官方训练/测试集，从训练序列末尾留出 H 点作验证，默认按选定轮数在完整训练集上重训。
- M4 六频率的 H 分别为 6/8/18/13/14/48，输入长度为 2H；预测还原到原始单位后计算 sMAPE、MASE 和基于 Naive2 的 OWA。
