# 当前开发阶段

本计划只描述新的 8×8 CNN/SNN → SNN → FPGA 主线。旧 28×28 CNN / AC620 阶段已经冻结，
索引见 [`history/README.md`](../history/README.md)。

## 阶段 1：统一器件编码与候选实验（进行中）

- [x] 保留 MLP latency baseline、large Conv-SNN 和 small Conv-IF-SNN 结果；
- [x] 将 `DeviceLatencyEncoder` 统一到 `experiments/common/`；
- [x] 建立可复现的 module execution 入口和轻量编码器契约测试；
- [ ] 明确后续实验的统一 seed、数据划分和记录模板。

## 阶段 2：matched 8×8 CNN baseline（下一步）

建立约 8k–12k 参数的 CNN 候选，与当前约 9,872 参数的 small Conv-IF-SNN 使用相同
8×8 预处理、55k/5k/10k 划分和 test-set-only-final-evaluation 规则。不得用 test
accuracy 做架构选择。

## 阶段 3：CNN/SNN 公平比较与 champion 选择

报告准确率、参数量、模型存储、稠密等效操作、有效事件/突触操作、激活/状态存储和
预计 FPGA BRAM/DSP/LUT 压力。当前不预先声称 SNN 获胜，也不把 GPU 时间当作 FPGA
性能或能耗证据。

## 阶段 4：正式模型与量化

仅在 champion 选定后建立真实的 `model/` 结构，冻结 architecture、checkpoint metadata、
量化方案、fixed-point/integer reference 和 hardware export。当前尚未开始。

## 阶段 5：Basys3 / Artix-7 硬件实现

仅在模型、位宽、状态存储和资源预算冻结后，在 `fpga/` 建立 RTL、仿真和 Vivado 工程，
再进行综合、实现与板级验证。当前仅保留规划说明，全部标记为 planned / not implemented。

## 当前禁止扩大范围

本阶段不重跑旧 CNN、不训练当前 SNN、不搜索超参数、不量化、不创建 RTL/Vivado 工程，
不进行综合、实现、bitstream、功耗或板级测试。
