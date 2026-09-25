# 8×8 `conv_small`：IF / LIF 纯软件消融

2026-09-25。沿用已冻结的 [8×8 SNN 配置](../conv_small/frozen_model.md)，只改两层隐藏神经元的膜电位更新。输入编码、网络参数、随机种子、优化器、读出、损失、数据划分与训练轮数保持一致。此目录是研究实验，不代表模型晋升或硬件验证。

## 变量与协议

隐藏层逐步执行 `u_t = decay · m_(t-1) + I_t`、`s_t = H(u_t - 1)`、`m_t = u_t - s_t`；训练使用原有 slope 5 surrogate，重置项的 spike 梯度仍截断。`decay=1` 为 IF；`decay<1` 为 LIF。仅隐藏膜泄露，输出读出的积分方式不变，泄露没有增加可训练参数。新模型同原模型均为 9,872 参数。同一组权重及 13 个输入上，`decay=1` 与原 IF 的 logits 逐元素完全相等（最大绝对差 `0.0`）。

- MNIST 训练前 55,000 / 验证后 5,000 / test 10,000；PIL bilinear 缩至 8×8，无数据增强。
- 器件 first-spike latency：24 步物理窗口量化至 `T=4`，`-1` 为 no-spike；复用训练集冻结的 30% 发放率阈值和 quantile 边界。
- 三个种子 7、17、27；20 epochs，batch 256；Adam，前 10 轮学习率 `1e-3`，后 10 轮 `3e-4`；梯度裁剪 1.0；时间权重 beta 0.5；事件正则系数 0.10，归一化常数 `20876.389236363637`。
- seed 17 仅根据验证准确率筛选 `decay ∈ {0.75, 0.90, 0.95}`；每次训练均以最高验证准确率选 checkpoint，同值取最早 epoch。筛选期间 test 评价次数为 0。选择 0.90 后扩展 seeds 7/27，选定的三个 checkpoint 各做一次最终 test。

## 验证筛选

| decay | seed 17 最佳 epoch | 最佳验证准确率 | 验证有效突触加法/图 | 筛选期 test 次数 |
|---:|---:|---:|---:|---:|
| 0.75 | 20 | 94.48% | 9,446.26 | 0 |
| **0.90** | **15** | **94.80%** | **9,212.97** | **0** |
| 0.95 | 20 | 94.74% | 9,216.62 | 0 |

复用冻结 IF seed 17 checkpoint 在同一验证集复测：95.20%，9,383.10 有效突触加法/图。新 LIF 候选未在验证准确率上超过该 IF 参照。

## 最终结果

| 种子 | IF 最佳验证 | IF test | LIF 最佳验证 | LIF test | LIF 隐藏脉冲/图 | LIF 有效突触加法/图 |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 94.96% | 93.47% | 94.80% | 93.26% | 297.02 | 9,750.01 |
| 17 | 95.20% | 93.60% | 94.80% | 93.47% | 284.24 | 9,260.94 |
| 27 | 94.76% | 93.33% | 94.68% | 92.90% | 297.13 | 9,597.09 |
| **均值** | **94.97%** | **93.47%** | **94.76%** | **93.21%** | **292.80** | **9,536.01** |

LIF test 样本标准差为 0.29 个百分点，IF 为 0.14 个百分点；三 seed 均值差（LIF − IF）为 **−0.26 个百分点**。IF 隐藏脉冲/图均值 301.45，有效突触加法/图均值 9,921.36。LIF 对应事件代理少约 3.88%，但有效突触加法是按有效连接数计的事件工作量代理，并非软件能耗、时延或 FPGA 功耗。三个种子的样本量较小；目前结果没有支持 LIF 在准确率上优于 IF。

IF 三 seed 数字来自既有[冻结记录](../conv_small/records/final_three_seed_record.md)，本轮未重训 IF。结构化聚合见[汇总](results/summary.json)；各次训练的 `run_config.json`、`history.csv`、`metrics.json`、checkpoint 及最终 test 记录位于 `results/decay_*`。

## 复现入口

从仓库根目录，使用 `D:\tools\anaconda3\envs\onn\python.exe`：

```powershell
# 验证筛选；其他 decay 用 0.75 或 0.95，各输出目录须全新
D:\tools\anaconda3\envs\onn\python.exe -m experiments.snn.lif_if.train --decay 0.9 --seed 17 --epochs 20 --time-steps 4 --input-preset t4_quantile_030 --readout-mode accumulated_membrane --temporal-beta 0.5 --event-lambda 0.10 --results-dir experiments/snn/lif_if/results/new_run

# 对已由验证集选定的 checkpoint 做一次最终 test
D:\tools\anaconda3\envs\onn\python.exe -m experiments.snn.lif_if.final_eval --checkpoint experiments/snn/lif_if/results/new_run/best_model.pt --decay 0.9 --seed 17 --output experiments/snn/lif_if/results/new_run/final_test.json

D:\tools\anaconda3\envs\onn\python.exe -m experiments.snn.lif_if.summarize
```

以上训练命令未带 `--eval-test`，训练模块只对验证集评价；最终 test 由独立入口运行。源文件 SHA256 记录在各训练目录的 `run_config.json`，checkpoint SHA256 记录在 `final_test.json`。本次运行未执行硬件流程。
