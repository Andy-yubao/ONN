# 8×8 CNN baseline（三 seed 训练与最终测试已完成）

本目录落实统一比较协议下的 MatchedConvSmall 候选，已完成冻结配置的三 seed 训练与最终测试。

目标是建立一个与当前轻量 SNN 参数量相近的 8×8 CNN baseline。当前参考 SNN 约
9,872 parameters，CNN 目标规模建议约 8k–12k；最终结构由实验决定。

未来 CNN 与 SNN 至少统一：

- MNIST 8×8 输入与相同 resize/preprocessing 原则；
- 55k train / 5k validation / 10k test；
- test set 只用于最终一次评价；
- 相同 seed policy；
- comparable parameter scale；
- 不用 test accuracy 做 architecture selection。

## 已冻结的阶段一协议

统一 seeds 为 `7, 17, 27`，顺序划分和共享数据入口见
[统一比较协议](../common/README.md)；新实验使用
[记录模板](../common/experiment_record_template.md)。CNN 阶段须复用
`experiments.common.comparison_protocol`，只依据验证集选择架构和 checkpoint。

## 首个候选

[架构设计与拟定训练规则](architecture.md)已落实为
[MatchedConvSmall](model.py)：两层 Conv/ReLU、Linear10，含 bias 共 9,930 参数。
前后向及参数预算检查已运行。架构和训练配置在测试前冻结，只按验证集选择 checkpoint。

## 运行

在仓库根目录使用 onn 环境，依次运行：

```powershell
D:\tools\anaconda3\envs\onn\python.exe -m experiments.cnn.train --seed 7 --device cuda
D:\tools\anaconda3\envs\onn\python.exe -m experiments.cnn.train --seed 17 --device cuda
D:\tools\anaconda3\envs\onn\python.exe -m experiments.cnn.train --seed 27 --device cuda
```

每个 seed 默认输出到 `results/seed_<seed>/`；拒绝覆盖非空目录，重现时用 `--results-dir` 指定新目录。
保存运行前的 `run_config.json`、逐轮 `history.csv`、最终 `metrics.json`、验证集曲线和
本地 `best_model.pt`。checkpoint 按仓库规则忽略。验证集预热后，每个 seed 仅进行一次完整测试。

## 三 seed 实测结果

最终测试准确率均值 **97.37%**，样本标准差 **0.11 个百分点**（ddof=1）。
完整配置、各 seed 结果和 SNN 对照见 [实验记录](three_seed_record.md)；
结构化结果见 [three_seed_summary.json](results/three_seed_summary.json)。硬件流程未运行，尚未选择正式 champion。
