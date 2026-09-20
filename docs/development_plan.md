# 当前开发阶段

本计划只描述新的 8×8 CNN/SNN → SNN → FPGA 主线。旧 28×28 CNN / AC620 阶段已经冻结，
索引见 [`history/README.md`](../history/README.md)。

## 阶段 1：统一器件编码与候选实验（已完成）

- [x] 保留 MLP latency baseline、large Conv-SNN 和 small Conv-IF-SNN 结果；
- [x] 将 `conv_small`（原 `snn_8x8_device_if_conv_small`）确认为当前最重要的 SNN 基线；
- [x] 将 `DeviceLatencyEncoder` 统一到 `experiments/common/`；
- [x] 建立可复现的 module execution 入口和轻量编码器契约测试；
- [x] 明确后续实验的统一 seed、数据划分和记录模板：`7, 17, 27`，顺序 55k/5k/10k，
  见 [统一比较协议](../experiments/common/README.md) 和
  [实验记录模板](../experiments/common/experiment_record_template.md)。

SNN 已完成同 seed 的 [三次重跑](../experiments/snn/conv_small/records/three_seed_record.md)，
最终测试均值 90.43%，样本标准差 0.20 个百分点；该结果不代表已完成 champion 选择。

## 阶段 2：matched 8×8 CNN baseline（首个 baseline 已完成）

建立约 8k–12k 参数的 CNN 候选，与当前最重要的 `conv_small` SNN 基线（约 9,872 参数）使用相同
8×8 预处理、55k/5k/10k 划分和 test-set-only-final-evaluation 规则。不得用 test
accuracy 做架构选择。

首个 [MatchedConvSmall 候选](../experiments/cnn/architecture.md) 已设计并实现，
含 bias 共 9,930 参数；前后向与参数预算检查已运行。冻结配置的三个 seed 训练与最终测试已完成，详见 [CNN 实验记录](../experiments/cnn/three_seed_record.md)。
最终测试均值 97.37%，样本标准差 0.11 个百分点；未做测试后调参。

## 阶段 3：CNN/SNN 公平比较与 SNN 验证冻结（已完成）

报告准确率、参数量、模型存储、稠密等效操作、有效事件/突触操作、激活/状态存储和
预计 FPGA BRAM/DSP/LUT 压力。当前不预先声称 SNN 获胜，也不把 GPU 时间当作 FPGA
性能或能耗证据。

已完成 [SNN 时间量化单 seed 探索](../experiments/snn/conv_small/records/time_quantization_record.md)：
固定 seed=17，对原 24 步 TTFS 做线性压缩到 T=12/8/5，其他配置不变；
其后完成 [输入时序编码联合实验](../experiments/snn/conv_small/records/input_encoding_joint_experiment.md)：
在 T=4/5/8 上联合比较 Linear/Quantile 和 baseline/约 30% input firing ratio；9 个新增配置均完成。
输入编码联合实验已完成。其后完成 [温和时间权重](../experiments/snn/conv_small/records/temporal_weighting_record.md)
与 [隐藏层事件正则化](../experiments/snn/conv_small/records/event_regularization_record.md)：固定 seed=17，
本轮方向筛选新增 2 个 beta 和 3 个 lambda，共 5 次正式训练；随后按用户追加请求对最终配置
做 1 次同 seed 确认运行，checkpoint 与逐轮非耗时指标完全复现。按 validation 规则选择 beta=0.5；在
validation 损失不超过 0.5 pct 的候选中选择事件成本最低的 lambda=0.10。该候选为
95.20% validation、93.60% test、9,440.04 有效突触加法/张。

最终配置已在 [frozen_model.md](../experiments/snn/conv_small/frozen_model.md) 冻结，新增 seed 7
和 27 后三 seed test 为 93.47% ± 0.14 pct；matched CNN 为 97.37% ± 0.11 pct。SNN 的
平均 effective synaptic additions 为 9,921.36/张，但该代理不能等同实测能耗。三种单调
器件曲线的 encoder-only 迁移共完成 9 次 inference，最大 mean accuracy drop 为 0.52 pct，
在当前模拟范围内支持重新校准前端后复用权重。完整结果见
[三 seed 记录](../experiments/snn/conv_small/records/final_three_seed_record.md) 与
[器件迁移记录](../experiments/snn/conv_small/records/device_curve_transfer_record.md)。

本阶段冻结的是最终 SNN 研究候选；CNN 仍有 3.90 pct mean accuracy 优势，真实器件行为、
FPGA 资源/功耗和总体 trade-off 尚未验证，因此未将任一方案 promote 为 production champion。

## 阶段 4：正式模型与量化

Frozen SNN 的 PTQ、动态范围分析和 fixed-point/integer reference 已先行建立在 `model/`，
作为 deployment candidate 验证；三 seed INT8 mean 为 93.24%，详见
[`quantization_record.md`](../model/snn/quantization_record.md)。production champion、供 RTL
使用的单一 checkpoint 和 hardware export 仍未冻结。

## 阶段 5：Basys3 / Artix-7 硬件实现

仅在模型、位宽、状态存储和资源预算冻结后，在 `fpga/` 建立 RTL、仿真和 Vivado 工程，
再进行综合、实现与板级验证。当前仅保留规划说明，全部标记为 planned / not implemented。

## 当前范围边界

旧 CNN / AC620 不作为当前主线重做；模拟单调器件曲线迁移和 frozen SNN 多 seed 稳健性已经
完成，Frozen SNN 的首轮 PTQ 与整数 reference 也已完成。反色输入、真实器件迁移、
RTL/Vivado 工程、综合、实现、bitstream、功耗和板级测试均未开始，不应在 champion 和
资源预算冻结前提前宣称已完成。是否执行其中某项以当前任务的明确范围为准。
