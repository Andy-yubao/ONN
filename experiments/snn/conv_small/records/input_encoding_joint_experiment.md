# SNN 输入时序编码联合实验记录

日期：2026-09-19。状态：9 个新增配置全部完成。范围来自
当时的 `ONN_SNN_joint_input_encoding_experiment_prompt.md`。本轮仅使用 seed=17，
不据此选择 production champion，也未开展 spike regularization、量化或 FPGA 工作。

## 冻结协议与实现

- 网络保持 Conv 1→16 / IF → Conv 16→32 stride=2 / IF → Linear 512→10，共 9,872 参数；
  IF threshold、subtract reset、surrogate gradient 和 accumulated-membrane readout 均未修改。
- 数据仍为顺序 train 55,000 / validation 5,000 / 官方 test 10,000，PIL bilinear Resize 8×8；
  20 epochs、batch 256、Adam、weight decay 0、gradient clip 1.0，前 10 轮 lr=1e-3，
  后 10 轮 3e-4。每轮只看 validation，最大值选 checkpoint，平局取最早轮次；
  加载最优 checkpoint 后 test 只评估一次。
- Linear 保持 `floor(old_t*T/24)`，`-1` 保持不发放。
- Quantile 只统计训练集 firing latency 直方图，按累计事件分位点把离散 latency 等级分成
  T 个连续组；同一 latency 等级不拆分，边界为各 bin 的 inclusive upper source latency。
  因此映射固定、全局且单调。离散 latency 的大质量点无法被拆开，bin 只能近似等量。
- 约 30% 配置直接从训练集 3,520,000 个 Resize 后像素取相邻强度等级中点，使 24 步窗口内
  firing count 最接近 30%，没有搜索 validation/test。最终固定
  `g_threshold=0.21026152308606838`；训练集实际为 **30.0866%**、19.2554 spikes/image。
  baseline `g_threshold=0.35` 在训练集实际为 **20.6399%**、13.2095 spikes/image。
- validation/test 不重新估计阈值或边界。同一配置的 train/validation/test 共用同一个编码器。

训练集确定的 quantile 边界如下：

| T | Baseline firing 边界 | 30% firing 边界 |
|---:|---|---|
| 4 | `[2, 3, 6]` | `[0, 1, 2]` |
| 5 | `[2, 3, 4, 6]` | `[0, 1, 2, 3]` |
| 8 | `[1, 2, 3, 4, 5, 6, 8]` | `[0, 1, 2, 3, 4, 5, 6]` |

## 完整联合结果

Actual firing ratio 使用与隐藏活动和事件工作量相同的官方 test 统计口径；训练集冻结值见上节。
三个 Linear + baseline 行直接引用既有正式记录和 metrics，未重训、未重新测试。

| T | Mapping | Target firing ratio | Actual firing ratio | Best Val Acc | Test Acc | MAC/image | L1 fire/IF/image | L2 fire/IF/image | Synaptic additions/image |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | Linear | baseline | 20.8445% | 92.36% | 91.07% | 352,256 | 0.1999 | 0.8442 | 20,738.68 |
| 4 | Quantile | baseline | 20.8445% | 92.42% | 91.04% | 352,256 | 0.1886 | 0.5479 | 18,435.01 |
| 4 | Linear | 30% | 30.1783% | 92.34% | 90.56% | 352,256 | 0.2101 | 0.7162 | 21,412.62 |
| 4 | Quantile | 30% | 30.1783% | **95.44%** | 93.88% | 352,256 | 0.2429 | 0.7028 | 23,758.61 |
| 5 | Linear | baseline | 20.8445% | 92.16% | 91.18% | 440,320 | 0.2197 | 1.0229 | 23,095.25 |
| 5 | Quantile | baseline | 20.8445% | 92.22% | 90.82% | 440,320 | 0.2060 | 0.6955 | 20,391.89 |
| 5 | Linear | 30% | 30.1783% | 93.28% | 91.26% | 440,320 | 0.2225 | 0.8963 | 23,276.29 |
| 5 | Quantile | 30% | 30.1783% | 94.96% | **93.98%** | 440,320 | 0.2554 | 0.8858 | 25,588.20 |
| 8 | Linear | baseline | 20.8445% | 93.38% | 91.41% | 704,512 | 0.2828 | 1.6583 | 30,794.64 |
| 8 | Quantile | baseline | 20.8445% | 91.92% | 90.44% | 704,512 | 0.2438 | 1.0731 | 25,057.82 |
| 8 | Linear | 30% | 30.1783% | 94.18% | 92.23% | 704,512 | 0.2733 | 1.5519 | 30,229.95 |
| 8 | Quantile | 30% | 30.1783% | 95.26% | 93.91% | 704,512 | 0.2719 | 1.3909 | 29,350.11 |

新增配置的最佳 epoch 依次为：T4 Quantile baseline 13、T4 Linear 30% 20、
T4 Quantile 30% 16、T5 Quantile baseline 20、T5 Linear 30% 17、
T5 Quantile 30% 17、T8 Quantile baseline 20、T8 Linear 30% 16、
T8 Quantile 30% 19。

## 输入时间直方图

以下均为官方 test 的平均 input spikes/image；每个像素最多发放一次。

| T | Mapping / ratio | 各时间步平均 spikes/image |
|---:|---|---|
| 4 | Linear / baseline | `[9.6276, 2.9674, 0.6254, 0.1201]` |
| 4 | Quantile / baseline | `[3.2746, 2.9809, 4.2494, 2.8356]` |
| 4 | Linear / 30% | `[17.1155, 1.6710, 0.3539, 0.1737]` |
| 4 | Quantile / 30% | `[2.0132, 8.4917, 4.8173, 3.9919]` |
| 5 | Linear / baseline | `[8.2439, 3.8732, 0.8566, 0.2467, 0.1201]` |
| 5 | Quantile / baseline | `[3.2746, 2.9809, 1.9884, 2.2610, 2.8356]` |
| 5 | Linear / 30% | `[16.3441, 2.0900, 0.7063, 0.1737, 0.0000]` |
| 5 | Quantile / 30% | `[2.0132, 8.4917, 3.0783, 1.7390, 3.9919]` |
| 8 | Linear / baseline | `[3.2746, 6.3530, 2.1199, 0.8475, 0.3787, 0.2467, 0.1201, 0.0000]` |
| 8 | Quantile / baseline | `[0.3273, 2.9473, 2.9809, 1.9884, 1.3837, 0.8773, 1.2426, 1.5930]` |
| 8 | Linear / 30% | `[13.5832, 3.5323, 1.1407, 0.5303, 0.3539, 0.0000, 0.1737, 0.0000]` |
| 8 | Quantile / 30% | `[2.0132, 8.4917, 3.0783, 1.7390, 1.0219, 0.7714, 0.4848, 1.7138]` |

Quantile 明显减少 Linear 的第一步拥挤，并填充 T=8 的全部时间步。30% 配置仍在第二步保留
8.4917 spikes/image 的峰值，因为该 source latency 本身包含大量相同事件；固定、单调映射不能在
不拆分相同 latency 的条件下进一步均分。

## 分析

### Quantile 的独立影响

在 baseline firing ratio 下，T4/T5 的 validation 仅各增加 0.06 个百分点，test 分别变化
-0.03/-0.36；T8 validation/test 分别下降 1.46/0.97。差异没有显示 Quantile 单独带来准确率收益。
另一方面，它把 synaptic additions 分别降低约 11.1%、11.7%、18.6%，并降低两层隐藏活动；
说明时间重分布改变了网络事件轨迹，但仅重分布原有 20.84% 输入信息不足以提高准确率。

### 提高 firing ratio 的独立影响

Linear 下从 baseline 提高到约 30%，T4 validation/test 变化 -0.02/-0.51，T5 为 +1.12/+0.08，
T8 为 +0.80/+0.82 个百分点。恢复弱笔画信息在 T5/T8 的 validation 上有正向迹象，但 T4 无收益。
事件成本没有随输入 spike 数简单同比增加，因为重新训练后的隐藏层活动同时变化。

### 组合协同

协同效应清晰：在约 30% firing ratio 下，Quantile 相对 Linear 的 validation/test 提升分别为：

- T4：+3.10 / +3.32 个百分点；
- T5：+1.68 / +2.72 个百分点；
- T8：+1.08 / +1.68 个百分点。

Quantile+30% 的三个配置都达到约 95% validation 和约 94% test，明显高于任一单独改动。
这支持“更多弱笔画信息需要配合更合理的时间展开才能发挥作用”，但仍只是单 seed 探索，
不能声称统计显著性或跨 seed 稳健性。

### 下一阶段候选

建议 **T=4 + Quantile + 30%** 进入下一阶段 hidden-activity / spike-regularization 实验：
它有最高 validation 95.44%，test 93.88%，同时 MAC 最低 352,256，synaptic additions 23,758.61。
T4/T5/T8 三个组合配置的 test 只相差 0.10 个百分点，T4 与 T8 的 validation 只差 0.18，
应描述为准确率接近；选择 T4 主要基于 validation 与计算/事件预算，而不是依据 test 挑选。
这不是 production champion 宣告。

## 输出、审计与异常

- 9 个正式目录均位于 `results/input_encoding_joint/`，各含 run_config、metrics、20 行 history、
  validation 最优 checkpoint 和训练曲线。审计确认：9/9 均为 20 epochs、seed=17、
  accumulated-membrane、test_evaluations=1；最佳 epoch 与 history 最大 validation 的最早轮次一致；
  checkpoint SHA256、参数量、MAC、input histogram 总数及训练集 calibration scope 均一致。
- 9 组使用完全相同的源码 SHA256 集合。三个既有 baseline 未重训。
- 代理最初执行第 2 组时应用户要求中止于 epoch 2；未做 test，输出保留在
  `T4_linear_030_interrupted_after_epoch2/`，不参与任何表格。用户随后用批量脚本从头完成正式组。
- 全仓 pytest 首次被既有无权限目录 `pytest-cache-files-to_lzb96/` 阻断在收集阶段；
  限定 `experiments/` 后测试通过。最终验证结果见完成报告。
- 单 seed 下小于约 0.5 个百分点的差异不解释为显著。所有 test 均只作一次最终参考，
  未据 test 追加配置或重训。
