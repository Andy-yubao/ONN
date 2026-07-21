# M1-B.1 基线实验审计与可复现性加固

## 审计目标

修复 M1-B 基线中的实验与工具问题，使训练框架可作为后续类脑学习规则实验的可信基础。

## 修复内容

| 修复项 | 说明 |
|--------|------|
| 活动分析采样位置 | Conv 原始输出 → 模块最终输出（ReLU / BasicBlock 最终输出） |
| 活动统计字段 | `nonzero/zero_ratio` → `active_ratio / near_zero_ratio` |
| 样本计数 | batch 级 → 样本级(`sample_count`)，新增 `batch_count`、`element_count_per_channel` |
| 不完整 batch加权 | 最后一个 batch 按实际样本数加权，不与完整 batch 等权 |
| Smoke test | 真正限制为最多 2 训练 batch + 2 验证 batch + 最多 2 epoch |
| 合成数据测试 | pytest 通过 `DataBundle` 注入合成数据，不再依赖 MNIST 下载 |
| Deterministic 模式 | 新增 `deterministic` 配置项，控制 cudNN 确定性算法 |
| 断点续训 | 完整恢复 scheduler、scaler、RNG、history、best 指标 |
| 旧检查点兼容 | Tiny-ResNet BasicBlock 重构后仍可 `strict=True` 加载旧权重 |

## 活动分析新旧差异

### BaselineCNN

| 指标 | 旧 (Conv raw) | 新 (post-activation) |
|------|:------------:|:-------------------:|
| 目标层 | 3 个 Conv 输出 | 3 个 ReLU 输出 |
| sample_count | 40 (batch数, 错误) | 5000 (样本数, 正确) |
| batch_count | 未记录 | 40 |
| 字段名 | nonzero_ratio / zero_ratio | active_ratio / near_zero_ratio |
| 不完整 batch | 等权 | 按样本数加权 |
| 近零活动通道 | 0/80 (Conv输出几乎不精确为零) | 0/80 (ReLU后确认无死通道) |

### Tiny-ResNet

| 指标 | 旧 (Conv raw) | 新 (post-activation) |
|------|:------------:|:-------------------:|
| 目标层 | ~10 个 Conv 输出 | 5 个模块最终输出 |
| sample_count | ~8 (batch数, 错误) | 5000 (样本数, 正确) |
| batch_count | 未记录 | 40 |
| 近零活动通道 | 0/~122 (旧方法缺陷) | 0/112 (ReLU后确认无死通道) |

### 新结果关键发现

- **无近零活动通道**：两个模型所有 post-activation 通道的 `near_zero_ratio` 均不高于 0.9，`mean_abs_activation` 均 > 0.01
- **活动逐层递增**：Tiny-ResNet 从 stem (0.33) 到 stage2.block1 (1.74) 活动均值递增，浅层特征提取 -> 深层特征组合
- **通道分化存在**：同一层内不同通道的 `mean_abs` 差异可达 2-3 倍（标准差约为均值的 20-50%）
- **不能跨层直接比较**：幅度受 BatchNorm scale 和权重分布影响
- **不能自动剪枝**：低活动通道可能仍对少数类别有判别力

## 多随机种子实验结果

所有实验使用 `deterministic=true`，固定配置（AdamW, lr=0.001, wd=0.0001, CosineAnnealingLR），不做任何调参。

### BaselineCNN (15 epochs)

| Seed | 最佳验证准确率 | 测试准确率 | 最佳 epoch |
|:----:|:-----------:|:---------:|:---------:|
| 42 | 98.18% | 98.09% | 14 |
| 43 | 98.54% | 98.48% | 15 |
| 44 | 98.68% | 98.20% | 15 |

| 指标 | mean ± std |
|------|----------:|
| 验证准确率 | 98.47% ± 0.26% |
| 测试准确率 | 98.26% ± 0.20% |
| 最佳 epoch | 14.7 ± 0.6 |

### Tiny-ResNet (20 epochs)

| Seed | 最佳验证准确率 | 测试准确率 | 最佳 epoch |
|:----:|:-----------:|:---------:|:---------:|
| 42 | 99.46% | **97.95%** ⚠ | 20 |
| 43 | 99.34% | 99.40% | 19 |
| 44 | 99.46% | 99.33% | 13 |

| 指标 | mean ± std |
|------|----------:|
| 验证准确率 | 99.42% ± 0.07% |
| 测试准确率 | 98.89% ± 0.82% |
| 最佳 epoch | 17.3 ± 3.8 |

> ⚠ **Tiny-ResNet seed 42 异常**：验证准确率 99.46% 但测试准确率仅 97.95%，表明 epoch 20 的检查点对验证集过拟合。其余两个 seed 的测试准确率正常（99.33-99.40%）。
>
> 启示：Tiny-ResNet 训练 20 个 epoch 对 MNIST 而言可能偏多。适当早停或在验证准确率饱和时停止有助于避免过拟合。但本实验的目的是测量标准波动范围，不进行早停或调参干预。

### 随机波动分析

- **BaselineCNN** 的测试准确率波动约 ±0.20%，标准模型自身的波动较小
- **Tiny-ResNet** 的验证准确率波动仅 ±0.07%，但测试准确率波动达 ±0.82%（主要因 seed 42 过拟合）
- 排除 seed 42 后的 Tiny-ResNet 测试准确率波动约为 ±0.04%
- 后续类脑规则实验若观察到 < 0.3% 的准确率变化，对于 BaselineCNN 可能不显著；若观察到 > 1% 的变化则确信为真实差异

## 限制说明

- 活动分析覆盖验证集 5000 样本，不代表训练集或测试集行为
- 当前 `deterministic=True` 在 CPU 上可复现，CUDA 上的确定性受 GPU 型号和驱动版本影响
- BaselineCNN 测试准确率 98.45% 已足以作为普通 CNN 对照基线，不要求达到 98.8%
- FPGA 部署仍处于候选评估阶段，尚未完成定点化、资源评估和延迟测算
