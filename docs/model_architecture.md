# MNIST 基线模型架构

## 任务定义

- **输入**：`1×28×28` 单通道灰度图（MNIST 手写数字）
- **输出**：10 类分类（数字 0-9）
- **评价指标**：分类准确率

---

## BaselineCNN

### 设计目的

作为无残差连接的普通 CNN 基线，轻量、硬件友好，便于与 Tiny-ResNet 结构对照。**BaselineCNN 是已完成的 INT8 FPGA 部署模型**（冻结，见 [data_format.md](../fpga/baseline_cnn/docs/data_format.md)）。

### 架构表

| 模块 | 层 | 输入 → 输出 | 参数量 | MACs |
|------|----|------------|-------:|-----:|
| Stem | Conv(3×3, 1→16, s=1, p=1) + BN + ReLU | 1×28×28 → 16×28×28 | 144 | 112,896 |
| | MaxPool(2×2) | 16×28×28 → 16×14×14 | — | — |
| Conv2 | Conv(3×3, 16→32, s=1, p=1) + BN + ReLU | 16×14×14 → 32×14×14 | 4,608 | 903,168 |
| | MaxPool(2×2) | 32×14×14 → 32×7×7 | — | — |
| Conv3 | Conv(3×3, 32→32, s=1, p=1) + BN + ReLU | 32×7×7 → 32×7×7 | 9,216 | 451,584 |
| Head | AdaptiveAvgPool(1) | 32×7×7 → 32×1×1 | — | — |
| | Flatten + Linear(32→10) | 32 → 10 | 330 | 320 |
| **总计** | | | **14,458** | **1,467,968** |

### 设计选择

- **无残差连接**：作为普通 CNN 基线对照
- **避免大型全连接层**：使用全局平均池化替代展平+大 FC
- **BatchNorm**：加速收敛，后续融合进卷积层以简化 FPGA 实现（**已完成 BN 融合**，见 [BN 融合实验](../experiments/model_deployment/baseline_cnn_bn_fusion/README.md)）
- **无 Dropout**：第一版保持最简单配置
- **无数据增强**：建立干净、易解释的基线

---

## Tiny-ResNet

### 设计目的

适合 MNIST 和资源受限 FPGA 的浅层残差网络，结构参考 ResNet 但大幅简化。**作为软件参考 / 高准确率备选**（尚未量化部署）。

### 为什么不是 ResNet-18

标准 ResNet-18 包含 4 个 stage（64→128→256→512 通道）和 7×7 大卷积核，对于 28×28 的 MNIST 输入严重过参数化。Tiny-ResNet 将通道缩小到 16→32，使用 3×3 卷积核，去除初始 MaxPool，更适合小尺寸输入和后续 FPGA 部署。

### 架构表

| 模块 | 层 | 输入 → 输出 | 参数量 | MACs |
|------|----|------------|-------:|-----:|
| Stem | Conv(3×3, 1→16, s=1, p=1) + BN + ReLU | 1×28×28 → 16×28×28 | 144 | 112,896 |
| Stage 1 | BasicBlock(16→16, s=1) × 2 | 16×28×28 → 16×28×28 | 9,216×2 | 1,806,336×2 |
| Stage 2 | BasicBlock(16→32, s=2) | 16×28×28 → 32×14×14 | 4,608+512 | 903,168+100,352 |
| | BasicBlock(32→32, s=1) | 32×14×14 → 32×14×14 | 18,432 | 1,806,336 |
| Head | AdaptiveAvgPool(1) + Linear(32→10) | 32×14×14 → 10 | 330 | 320 |
| **总计** | | | **42,938** | **13,761,088** |

### BasicBlock 结构

```text
输入
  ├── Conv(3×3, stride=s) → BN → ReLU → Conv(3×3) → BN ──┐
  └── Shortcut: Identity / Conv(1×1, stride=s) → BN ──────┘
                            ↓
                        ReLU → 输出
```

- 输入输出尺寸一致时：捷径为恒等映射
- 通道或尺寸变化时：捷径为 `Conv(1×1) + BN`

### 设计选择

| 选择 | 原因 |
|------|------|
| 全局平均池化 | 避免大 FC 层，减少参数量，适应可变输入尺寸 |
| BatchNorm | 加速训练稳定，后续可融合进卷积 |
| 16→32 通道 | 保持低参数量，适合 FPGA |
| 2 个 stage 而非 4 个 | MNIST 复杂度低，不需要过深网络 |
| 无 7×7 卷积 | 28×28 输入下 3×3 感受野已足够 |

---

## 已完成的部署链路（BaselineCNN）

> 从 FP32 到 AC620 板级自检的完整链路均已落地，见 [硬件目标](../fpga/baseline_cnn/docs/hardware_target.md)。

| 阶段 | 结果 | 证据 |
|------|------|------|
| 浮点基线 | seed 42：98.45%（BaselineCNN）/ 99.43%（Tiny-ResNet） | [m1_baselines](../experiments/model_baselines/m1_baselines/README.md) |
| 多种子审计 | BaselineCNN test 98.26%±0.20；TinyResNet 98.89%±0.82（seeds 42/43/44） | [multiseed](../experiments/model_baselines/m1_baseline_audit/README.md) |
| BN 融合 | 融合后 98.48%，预测 100% 一致 | [bn_fusion](../experiments/model_deployment/baseline_cnn_bn_fusion/README.md) |
| INT8 PTQ | 方案 A 98.46%（降 0.02pp） | [int8_ptq](../experiments/model_deployment/baseline_cnn_int8_ptq/README.md) |
| 纯整数参考 | 98.47%，GAP 前逐位一致 | [int8_reference](../experiments/model_deployment/baseline_cnn_int8_reference/README.md) |
| 参数导出 | `.mem`/`.mif`/`.vh` + SHA256 checksums | [params](../fpga/baseline_cnn/params/) |
| 完整 RTL + 仿真 | 11 节点黄金 trace + 10 smoke 逐位一致 | [rtl_microarchitecture](../fpga/baseline_cnn/docs/rtl_microarchitecture.md) |
| 50 MHz STA | 最小 Fmax 56.41 MHz | [hardware_target](../fpga/baseline_cnn/docs/hardware_target.md) §12 |
| 固定 digit8 板级自检 | prediction=8，PASS，JTAG SRAM | [hardware_target](../fpga/baseline_cnn/docs/hardware_target.md) §12.1 |

### 部署现状总结

BaselineCNN 已具备 INT8 定点硬件部署资格（INT32 累加无溢出、无回绕、确定性、逐层 bit-accurate），并在 AC620 V2 上完成**固定 digit8、片上 ROM 输入**的板级自检。以下事项**尚未完成**，不能扩大为"完整系统部署全部完成"：

- 外部任意图像输入 / UART 图像传输
- 测试集级硬件准确率（≥1000 张）
- 功耗或单次推理能耗实测
- EPCS Flash 固化
- 完整最终产品化接口

---

## 后续研究方向（2026-08-02 组会）

- 器件模型驱动训练：标准 RGB 图像 → 电导/电流表示 → 训练与评价
- 定向 Hebbian 连接增强
- KAN 候选
- 多次采样与时序特征编码

以上均处于决策/待设计阶段，参数（数据集、矩阵尺寸、公式、结构）为开放问题，见 [open_questions.md](open_questions.md) 的 NQ 清单。

---

## 本阶段不实现的特性

- ❌ 定向 Hebbian 更新规则的数学实现（公式 TBD）
- ❌ KAN 实现（结构 TBD）
- ❌ 4×4 光电阵列数据实时接入（不要求实时耦合）
- ❌ FPGA 在线训练
- ❌ 外部图像输入与测试集级硬件评价（M3-B.4 未完成）

> 注：量化、整数参考、RTL 与 bit-accurate 比对**均已实现**，不再属于"不实现"列表。

---

## 后续扩展接口

- `model/onn_model/activity.py` — 通道活动统计，可用于活动依赖衰减规则
- `model/onn_model/engine.py` —— `train_one_epoch` 可替换优化器为自定义更新规则
- `model/onn_model/int8_reference.py` — 纯整数参考模型（冻结数值标准）
- `model/export_baseline_cnn_hardware.py` — 硬件参数导出入口
