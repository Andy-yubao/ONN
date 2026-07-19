# 论文笔记 04：CNN/SNN FPGA 定量比较

## 论文信息

- **标题**：To Spike or Not to Spike: A Quantitative Comparison of SNN and CNN FPGA Implementations
- **论文编号**：P4
- **证据等级**：C（FPGA 算法参考）

## 纳入原因

1. 同一平台上定量比较 CNN 和 SNN 的 FPGA 实现
2. 定点 CNN 架构设计（流式数据流、PE、FIFO、滑窗）
3. 关键发现：小任务 SNN 无稳定优势

## 核心内容

### 硬件平台
- PYNQ-Z1 (XC7Z020) 和 ZCU102
- **均为 Xilinx，不是 Cyclone IV**

### 关键设计
- 流式数据流流水线
- 滑窗单元（行缓冲）
- PE 阵列 + SIMD 可配置
- 片上权重 BRAM/LUTRAM

### 关键发现
- MNIST 级别小任务：SNN 延迟和能效不具优势
- SNN 优势需任务规模大、输入稀疏（SVHN/CIFAR-10）
- SNN 开销：膜电位存储、事件队列管理

## 可复用设计
- 流式数据流架构思想
- 滑窗单元和 PE 阵列概念
- 混合精度量化思路

## 不可移植
- FINN/Brevitas/ONNX → Xilinx，非 Cyclone IV
- BRAM（Xilinx 36Kb vs Altera M9K）不同
- 所有资源数据是 Xilinx 结果

## 与 4×4 关系
- 支持小型定点 ANN/CNN 为第一版基线
- SNN 作为条件分支

## 关键章节
- Sec. III: CNN 架构
- Sec. IV: SNN 架构
- Sec. V: 定量比较和讨论
