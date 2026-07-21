# M1-B 基线训练实验结果

> **⚠ 审计说明**：该实验为 M1-B（初始基线）结果，活动分析统计的是 Conv 原始输出（pre-activation），已被 M1-B.1 审计后的 post-activation 分析取代。详见 `../m1_baseline_audit/`。

## 实验目标

建立 MNIST 手写数字分类的标准基线：普通 CNN（BaselineCNN）与浅层残差网络（Tiny-ResNet），为后续类脑学习规则研究和 FPGA 部署候选提供可靠基准。

## 数据划分

| 划分 | 样本数 |
|------|-------:|
| 训练集 | 55,000 |
| 验证集 | 5,000 |
| 测试集 | 10,000（官方） |

## 训练配置

| 参数 | 值 |
|------|-----|
| 损失函数 | CrossEntropyLoss |
| 优化器 | AdamW |
| 学习率 | 0.001 |
| 权重衰减 | 0.0001 |
| 调度器 | CosineAnnealingLR |
| Batch size | 128 |
| 随机种子 | 42（单种子，仅作参考） |
| 确定性模式 | 未启用 |
| 数据预处理 | ToTensor + Normalize(0.1307, 0.3081) |

## 运行环境

| 项目 | 值 |
|------|-----|
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU |
| CUDA | 12.8 |

## 实验结果

| 模型 | Epochs | 最佳验证准确率 | 最佳验证 Loss | 测试准确率 | 测试 Loss | 参数量 | MACs |
|------|:-----:|:------------:|:-----------:|:---------:|:---------:|:-----:|:----:|
| BaselineCNN | 15 | **98.20%** | 0.0589 | **98.45%** | 0.0538 | 14,458 | 1.47M |
| Tiny-ResNet | 20 | **99.40%** | 0.0199 | **99.43%** | 0.0167 | 42,938 | 13.76M |

**注意**：以上结果为单种子（seed=42）且未启用 deterministic 模式的结果。后续多种子对照实验见 `../m1_baseline_audit/multiseed_results.json`。

## 模型分析

### BaselineCNN（14,458 参数，1.47M MACs）

- 3 个 Conv-BN-ReLU 块 + AdaptiveAvgPool + Linear
- 轻量、无残差连接，为低资源 FPGA 候选
- 最大中间激活：12,544 元素

### Tiny-ResNet（42,938 参数，13.76M MACs）

- Stem + 2 个 BasicBlock 阶段 + AdaptiveAvgPool + Linear
- 残差连接，16→16→32 通道渐进
- 参数量约为 BaselineCNN 的 3 倍，MAC 约 9.4 倍
- 最大中间激活：12,544 元素（与 BaselineCNN 相同）

## 通道活动分析摘要（Conv 原始输出，已废弃）

> ⚠ 以下分析基于 Conv 原始输出（pre-activation），已被更准确的 post-activation 分析取代。
> 此处仅保留用于审计追溯。

### BaselineCNN

- 所有卷积输出通道均有非零活动（原始 Conv 输出几乎不会精确为零）
- 各层通道活动存在中等差异（标准差约为均值的 55-70%）
- Stem 层活动差异最大，conv3 层活动最均匀

### Tiny-ResNet

- 所有通道原始 Conv 输出均有非零活动
- 深层激活均值更高（stage2.1.conv1 达 3.20 vs stem 的 0.28）
- Stage 2 活动差异大于 Stage 1

### 限制说明

Conv 原始输出不经过 ReLU 处理，因此：
- **不能** 通过 `zero_ratio` 判断死通道
- **不能** 直接作为"使用程度"或"神经元活动程度"指标
- 不同卷积层的幅度不可直接比较（受各自权重范数影响）

## 后续研究建议

1. Tiny-ResNet 的 99.43% 测试准确率接近 MNIST 饱和性能，适合作为后续类脑规则的主要算法研究模型
2. BaselineCNN 测试准确率 98.45% 已足以作为普通 CNN 对照基线（不要求达到 98.8%）
3. 训练损失总体持续下降，验证损失存在正常波动
4. 两个模型均具备进一步进行定点化和 FPGA 资源评估的条件
5. 活动分析接口已可用于设计通道级规则，但需基于 post-activation 结果
6. **本实验仅使用单随机种子**，标准模型自身的随机波动幅度需通过多种子实验确定（参见审计实验）
