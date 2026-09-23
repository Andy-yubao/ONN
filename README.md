# ONN — Device-Coded 8×8 SNN Research and FPGA Deployment

当前主线：

```text
器件动态建模
→ 8×8 first-spike latency encoding
→ matched CNN/SNN experiments
→ SNN champion selection
→ quantization
→ FPGA deployment
```

ONN 当前研究重点是把 8×8 输入、器件动态响应和时序脉冲编码放进同一实验协议，公平
比较参数量相近的 CNN 与 SNN。若 SNN 在准确率、计算稀疏性、器件编码耦合和 FPGA 资源
之间形成可接受折中，才会 promote 到 `model/` 并进入硬件实现。

## 当前阶段结果

当前验证阶段配置已经冻结：8×8、T=4、Quantile、约 30% input firing、beta=0.5、
lambda=0.10，网络共 9,872 参数。正式三 seed held-out test 为 **93.47% ± 0.14 pct**；
参数量相近的 matched CNN（9,930 参数）为 **97.37% ± 0.11 pct**。完整对比见
[Frozen SNN 三 seed 记录](experiments/snn/conv_small/records/final_three_seed_record.md)。

三种简单单调器件曲线的 encoder-only 迁移也已完成：保持约 30% firing ratio 并重算
Quantile boundaries 后，平均准确率相对原器件下降 0.00–0.52 pct，在该模拟范围内支持
“换器件 → 重校准 encoder → 复用冻结 SNN 权重”。这不是实际器件或 FPGA 证据，详见
[器件曲线迁移记录](experiments/snn/conv_small/records/device_curve_transfer_record.md)。Frozen
SNN 的 PTQ 与整数 reference 已完成：三 seed INT8 mean 为 **93.24% ± 0.17 pct**，比 FP32
下降 0.22 pct，详见[量化记录](model/snn/quantization_record.md)。seed 17 deployment
checkpoint 的事件驱动稀疏 SNN RTL 已通过仿真级逐 bit 对拍。Basys3 USB-UART wrapper
和 PC host 已完成无板仿真；综合与实现结论见 [`fpga/basys3/`](fpga/basys3/) 的记录。
首次真实 Basys3 验证的 4 个人工输入及 100 张 validation 样本均与整数参考逐项一致，
详见[板级验证记录](fpga/basys3/verification_record.md)。当前仍未 promote 为项目
production champion，也未完成真实器件输入验证或实测功耗。

## 硬件方向

- Legacy platform：AC620 / Cyclone IV；
- Current candidate target：Basys3 / Artix-7；
- seed 17 SNN 的部署数值契约已冻结；Basys3 实现报告已确认 25 MHz 内部时钟下的资源与静态时序；
- 当前单 lane sparse RTL、PC/FPGA UART 通信和脚本化 Vivado 流程用于部署可行性验证。

## 目录职责

```text
experiments/   候选网络、失败/消融实验和 CNN/SNN 公平比较
model/         冻结部署候选、未来正式 champion、量化、整数参考和硬件导出
fpga/          冻结部署候选的 FPGA 验证与未来正式模型实现
history/       历史阶段索引与 Git tag 入口，不复制旧源码
docs/          当前范围、架构、阶段计划和开放问题
```

当前保留的三项 SNN 研究资产位于；其中 `conv_small` 是后续 CNN/SNN 公平比较和模型
选择的优先参考基线：

- [`experiments/snn/mlp_latency_baseline/`](experiments/snn/mlp_latency_baseline/)；
- [`experiments/snn/conv_large/`](experiments/snn/conv_large/)；
- [`experiments/snn/conv_small/`](experiments/snn/conv_small/)（原
  `snn_8x8_device_if_conv_small`，最重要的 SNN 基线）。

开始阅读：[`docs/project_scope.md`](docs/project_scope.md)、
[`docs/system_architecture.md`](docs/system_architecture.md)、
[`experiments/README.md`](experiments/README.md)。
