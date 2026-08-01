# BaselineCNN BatchNorm 融合验证报告

> 阶段：FP32 模型交付准备 —— BatchNorm 融合与正确性验证
> 日期：2026-07-31
> 范围：仅处理 BaselineCNN seed 43 候选；不含 TinyResNet、不含整数定点量化

## 1. 源 checkpoint

| 项 | 值 |
|---|---|
| checkpoint 路径 | `model/runs/BaselineCNN/20260721_163832/best.pt` |
| 模型 | `BaselineCNN` |
| seed | 43 |
| 训练 epoch | 15（最佳 epoch 15） |
| deterministic | true |
| 训练时 git commit | `0d45865` |
| 历史最佳验证准确率 | 0.9854 |
| 重算测试准确率 | 0.9848 |

加载方式：`strict=True`，当前 HEAD（`8561c2a`）下全部权重键匹配，无缺失/多余键。

## 2. 模型结构

### 原始 BaselineCNN（融合前）

```text
stem :  Conv2d(1→16, 3×3, bias=False) → BatchNorm2d(16) → ReLU
pool1:  MaxPool2d(2×2)
conv2:  Conv2d(16→32, 3×3, bias=False) → BatchNorm2d(32) → ReLU
pool2:  MaxPool2d(2×2)
conv3:  Conv2d(32→32, 3×3, bias=False) → BatchNorm2d(32) → ReLU
pool :  AdaptiveAvgPool2d(1)
fc   :  Linear(32→10)
```

输入 `[N, 1, 28, 28]`，输出 `[N, 10]`。

### 融合后（无 BatchNorm）

```text
stem :  Conv2d(1→16, 3×3, bias=True) → ReLU
pool1:  MaxPool2d(2×2)
conv2:  Conv2d(16→32, 3×3, bias=True) → ReLU
pool2:  MaxPool2d(2×2)
conv3:  Conv2d(32→32, 3×3, bias=True) → ReLU
pool :  AdaptiveAvgPool2d(1)
fc   :  Linear(32→10)
```

输入输出形状保持不变。

## 3. BN 融合公式

对每个 `Conv2d → BatchNorm2d` 对（eval 语义，使用 `running_mean` / `running_var`）：

```text
scale        = gamma / sqrt(running_var + eps)
W_fused      = W * scale                          # 按输出通道缩放
b_fused      = (b - running_mean) * scale + beta  # 原始卷积无 bias 时取 b = 0
```

融合后的卷积保留原始卷积的 `kernel_size`、`stride`、`padding`、`dilation`、`groups`，并携带融合后的 `bias`。

## 4. 融合前后模块数量

| 模块类型 | 原始模型 | 融合模型 |
|---|---|---|
| BatchNorm2d | 3 | **0** |
| Conv2d | 3 | 3 |
| Linear | 1 | 1 |

融合只作用于 `stem` / `conv2` / `conv3` 三个顶层 `Sequential`（`Conv2d → BN → ReLU` 模式），`pool1/pool2/pool/fc` 不变。

## 5. 逐层误差表（固定 256 张 MNIST 测试图片，无 shuffle）

| 层 | 输出 shape | 最大绝对误差 | 平均绝对误差 | MSE | NaN/Inf |
|---|---|---|---|---|---|
| stem_relu | [256, 16, 28, 28] | 1.43e-06 | 2.36e-08 | 4.14e-15 | 无 |
| pool1 | [256, 16, 14, 14] | 1.43e-06 | 3.57e-08 | 7.51e-15 | 无 |
| conv2_relu | [256, 32, 14, 14] | 2.86e-06 | 6.87e-08 | 2.24e-14 | 无 |
| pool2 | [256, 32, 7, 7] | 2.62e-06 | 1.11e-07 | 4.05e-14 | 无 |
| conv3_relu | [256, 32, 7, 7] | 1.91e-05 | 9.09e-07 | 2.46e-12 | 无 |
| pool | [256, 32, 1, 1] | 3.10e-06 | 4.20e-07 | 3.27e-13 | 无 |
| logits | [256, 10] | 9.30e-06 | 2.29e-06 | 7.88e-12 | 无 |

最大误差出现在 `conv3_relu`（1.91e-05），仍远低于验收阈值 1e-4，为浮点运算顺序差异所致。

## 6. 完整测试集准确率（10,000 张）

| 模型 | 测试准确率 |
|---|---|
| 原始 BaselineCNN | **0.984800** |
| 融合后 BaselineCNN | **0.984800** |
| 准确率差值 | **0.0000（0.0000 pp）** |

## 7. 预测一致率

| 指标 | 值 |
|---|---|
| 一致样本数 | 10000 / 10000 |
| 不一致样本数 | 0 |
| 预测一致率 | **100.0000%** |

## 8. logits 误差（完整测试集）

| 指标 | 值 |
|---|---|
| 最大 logits 绝对误差 | **1.049042e-05** |
| 平均 logits 绝对误差 | **2.220027e-06** |

## 9. 融合模型本地路径

```text
model/artifacts/baseline_cnn_seed43_bn_fused_fp32.pt
```

- 由 `torch.save` 保存，`*.pt` 已 gitignore，仅本地。
- 结构（dict）：`model_name`、`source_checkpoint`、`source_seed`、`bn_fused`、`dtype`、`input_shape`、`state_dict`、`original_test_accuracy`、`fused_test_accuracy`、`max_abs_error`、`prediction_agreement`。
- 原 `best.pt` 与原始带 BN 模型均未覆盖、未删除。

### 融合模型加载方法

融合模型结构与原始 `BaselineCNN` 不同（无 BN、conv 带 bias），**不能**用 `BaselineCNN().load_state_dict()` 直接加载。使用 `quantization.py` 提供的辅助函数：

```python
from onn_model.quantization import load_bn_fused_model
from onn_model.models.baseline_cnn import BaselineCNN

fused, meta = load_bn_fused_model(
    "model/artifacts/baseline_cnn_seed43_bn_fused_fp32.pt",
    BaselineCNN,                       # 模型工厂：用未融合类新建结构
    torch.device("cpu"),
)
```

`load_bn_fused_model` 会先对全新 `BaselineCNN()` 执行 `fuse_model_bn` 重建融合结构，再以 `strict=True` 加载 `state_dict`。已通过保存→重载→固定样本推理逐位一致验证（round-trip 输出完全一致）。

## 10. 当前结论

1. 现有 BN 融合实现（`fuse_conv_bn_eval` / `fuse_model_bn` / `check_bn_fusion_error`）经审查公式与替换逻辑正确，**无需修复**。本阶段新增 `load_bn_fused_model` 作为融合模型的官方加载入口。
2. BN 融合为数学等价变换，实测数值误差远小于 1e-4（最大 logits 误差 1.05e-05），测试准确率完全不变（0.9848），预测结果 100% 一致。
3. 融合后模型为纯 `Conv2d + ReLU + Pool + Linear` 结构，无 BatchNorm，是后续量化的合适基础。
4. 验收标准全部满足：
   - strict=True 加载 ✅
   - 融合后 BatchNorm2d 数量 = 0 ✅
   - 原始模型未被修改 ✅
   - 所有对应层输出 shape 一致 ✅
   - 所有输出无 NaN/Inf ✅
   - 固定样本 logits 最大误差 9.30e-06 < 1e-4 ✅
   - 完整测试集预测一致率 100.00% ≥ 99.99% ✅
   - 融合后准确率下降 0.0000 pp ≤ 0.01 pp ✅
5. 自动化测试：新增 `model/tests/test_bn_fusion.py`（11 项，合成数据、CPU、不依赖 MNIST 下载），全套 115 项测试通过。

## 11. 尚未完成（后续阶段）

以下内容**未**在本阶段完成，FP32 BN 融合模型**不等同于** FPGA 部署就绪：

- [ ] 激活值范围校准（activation calibration）
- [ ] INT8（或 INT16）整数定点量化
- [ ] 定点参考模型（fixed-point reference model）
- [ ] FPGA 参数导出（权重/偏置/scale 二进制导出）
- [ ] RTL 实现
- [ ] bit-accurate 验证（FPGA 与软件参考逐位一致）

## 复现

```bash
# 验证 + 导出融合模型（默认 seed 43 checkpoint）
python model/verify_bn_fusion.py
# 或指定 checkpoint
python model/verify_bn_fusion.py --checkpoint model/runs/BaselineCNN/20260721_163832/best.pt

# 自动化测试
python -m pytest model/tests -q
```
