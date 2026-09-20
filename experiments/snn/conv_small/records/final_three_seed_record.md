# Frozen T=4 SNN 三 seed 正式结果

日期：2026-09-19。配置见 [frozen_model.md](../frozen_model.md)。本轮只新增 seed 7 和
seed 27 两次 20-epoch 训练；seed 17 复用既有正式结果，没有重训。每个 seed 均按 validation
选择 best checkpoint，final test 只执行一次。

## SNN 结果

| Seed | Best epoch | Best Val | Test | L1 fire/IF/image | L2 fire/IF/image | Hidden spikes/image | Synaptic additions/image |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 17 | 94.96% | 93.47% | 0.0734 | 0.4426 | 301.8018 | 10,141.29 |
| 17 | 18 | 95.20% | 93.60% | 0.0639 | 0.4388 | 290.0512 | 9,440.04 |
| 27 | 17 | 94.76% | 93.33% | 0.0700 | 0.4704 | 312.4860 | 10,182.75 |
| **Mean** | — | **94.97%** | **93.47%** | **0.0691** | **0.4506** | **301.4463** | **9,921.36** |

- Test：**93.47% ± 0.14 pct**（sample std）。
- Best Val：**94.97% ± 0.22 pct**（sample std）。
- test input firing ratio 三次均为 `30.17828125%`；encoder 与 frozen 配置一致。
- 三个 checkpoint SHA256 分别为
  `44bf2121…bdbf0431`、`ecd4fa94…8c845e6`、`ffd273a8…d6b685e`。

结构化产物见 [`results/final_three_seed/summary.json`](../results/final_three_seed/summary.json)
及各 seed 目录。Hidden spikes/image 是 L1 与 L2 spike events/image 之和；effective
synaptic additions 是事件驱动潜在工作量代理，不是软件或 FPGA 实测能耗。

## Matched CNN 对比

CNN 数据来自既有 [三 seed 正式记录](../../../cnn/three_seed_record.md)，未重训或重测。

| Metric | CNN | Frozen SNN |
|---|---:|---:|
| Parameters | 9,930 | 9,872 |
| Test accuracy mean | 97.37% | 93.47% |
| Test accuracy sample std | 0.11 pct | 0.14 pct |
| Dense MAC/image | 88,064 | 352,256 |
| Time steps | 1 | 4 |
| Effective synaptic additions/image | N/A | 9,921.36 |
| Input event sparsity | N/A | 30.09% train firing ratio |
| Hidden activity | N/A | L1 0.0691 / L2 0.4506 fire/IF/image |

CNN 比 Frozen SNN 的三-seed test mean 高 **3.90 pct**。CNN 的 88,064 是普通 dense
MAC/image；SNN 的 352,256 是四个时间步的 dense-equivalent MAC，并非实际事件数。
9,921.36 effective synaptic additions/image 是按有效连接计数的事件驱动工作量代理，不能
直接换算为当前软件能耗或 FPGA 功耗；FPGA 实测功耗尚未获得。因此结果支持 CNN 的准确率
优势和 SNN 的事件稀疏性，但不支持仅凭代理计数宣称 SNN 整体更优。

