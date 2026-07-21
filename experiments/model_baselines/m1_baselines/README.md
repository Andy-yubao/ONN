# M1-B 基线训练实验结果

## 实验目标

建立 MNIST 手写数字分类的标准基线：普通 CNN（BaselineCNN）与浅层残差网络（Tiny-ResNet），为后续类脑学习规则研究和 FPGA 部署提供可靠基准。

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
| 随机种子 | 42 |
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

## 模型分析

### BaselineCNN（14,458 参数，1.47M MACs）

- 3 个 Conv-BN-ReLU 块 + AdaptiveAvgPool + Linear
- 轻量、无残差连接，适合 FPGA 部署
- 最大中间激活：12,544 元素

### Tiny-ResNet（42,938 参数，13.76M MACs）

- Stem + 2 个 BasicBlock 阶段 + AdaptiveAvgPool + Linear
- 残差连接，16→16→32 通道渐进
- 参数量约为 BaselineCNN 的 3 倍，MAC 约 9.4 倍
- 最大中间激活：12,544 元素（与 BaselineCNN 相同）

## 通道活动分析摘要

### BaselineCNN

- 所有卷积输出通道均活跃（无死通道）
- 各层通道活动存在中等差异（标准差约为均值的 55-70%）
- Stem 层活动差异最大，conv3 层活动最均匀
- 无 channels 的 mean_abs < 0.01

### Tiny-ResNet

- 所有通道均活跃，无死通道
- 深层激活均值更高（stage2.1.conv1 达 3.20 vs stem 的 0.28）
- Stage 2 活动差异大于 Stage 1，反映出高级特征差异化
- 捷径分支（shortcut）活动适中，正常参与信息传递

## 后续研究建议

1. Tiny-ResNet 的 99.43% 测试准确率已接近 MNIST 饱和性能，更适合作为后续主模型
2. 标准训练管线可靠，损失单调下降，无 NaN/Inf
3. 当前模型规模适合 FPGA 部署（< 50K 参数，< 14M MACs）
4. 活动分析显示无死通道，类脑规则设计可尝试活动依赖衰减或活动调制梯度
5. BaselineCNN 测试准确率 98.45% 略低于初步目标（98.8%），可通过增加 epoch 或微调学习率提升
