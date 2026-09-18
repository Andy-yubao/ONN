# 当前研究要求

## 数据与编码

- 统一使用 MNIST 8×8 输入和一致的 resize/preprocessing 原则；
- 器件编码采用共同的 `DeviceLatencyEncoder`；
- `G0=0.10`、`alpha=0.90`、`tau=5.0`、`G_threshold=0.35`、`T=24` 不在公平比较中
  随意改变；
- 无法达到阈值的像素以 first-spike time `-1` 表示不发放。

## 公平实验

- 训练/验证/测试为 55,000 / 5,000 / 10,000；
- test set 只用于最终评价，不用于架构选择、early stop 或 checkpoint 选择；
- CNN 与 SNN 使用相同 seed policy 和相近参数量；
- 同时记录准确率、参数/存储、稠密等效工作量、有效事件/突触工作量和状态存储；
- 不以 GPU inference time 推断 FPGA 延迟或功耗。

## Promotion 与硬件

- 未完成公平比较前，候选只能留在 `experiments/`；
- 只有正式 champion 才能进入 `model/`；
- 量化、整数参考和硬件导出必须来自冻结的正式模型；
- Basys3 / Artix-7 只是当前候选平台，正式 RTL、Vivado 工程、综合和板级验证待后续阶段；
- 当前不编造硬件资源、时序、功耗或板级准确率结论。
