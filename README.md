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

以下数字来自已提交实验记录，本次架构重构未重新计算：

- 冻结的 28×28 CNN 无法直接兼容 8×8/4×4 低有效分辨率输入；
- MLP latency SNN：87.15%；
- large Conv-SNN architecture exploration：91.27%，不作为严格独立 held-out test benchmark；
- 当前最重要的 SNN 基线是 small Conv-IF-SNN（原实验目录
  `snn_8x8_device_if_conv_small`）：held-out test 90.22%，9,872 parameters；
- 95% 目标尚未达到；
- matched 8×8 CNN baseline 尚未完成；
- 尚未选出 production SNN；
- 尚未开始当前 SNN 的量化和 Basys3 RTL。

## 硬件方向

- Legacy platform：AC620 / Cyclone IV；
- Current candidate target：Basys3 / Artix-7；
- hardware target 尚未最终 freeze；
- 在 champion、数据位宽、状态存储和资源预算冻结前，不进入正式 RTL。

## 目录职责

```text
experiments/   候选网络、失败/消融实验和 CNN/SNN 公平比较
model/         正式 champion、量化、整数参考和硬件导出
fpga/          正式 champion 的 FPGA 实现
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
