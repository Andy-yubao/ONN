# BaselineCNN INT8 PTQ 可行性评估报告

> 阶段：软件 W8A8 量化仿真 —— 融合后模型的 INT8 训练后量化（PTQ）可行性评估
> 日期：2026-07-31
> 范围：仅处理 BN 融合后的 BaselineCNN（seed 43 候选）；不含 TinyResNet、不含 M2 紧凑模型、不含 QAT、不含 FPGA bit-accurate

**结论：INT8 W8A8 可行。** 方案 A（权重 per-tensor、激活 per-tensor，最简单的硬件方案）测试准确率 98.46%，相对 FP32（98.48%）仅下降 **0.02 个百分点**，满足验收标准（≤0.2pp）。按"优先选择满足要求的最简单方案"原则，**推荐方案 A** 作为后续整数参考模型的基础。

## 1. 输入模型与校准数据

| 项 | 值 |
|---|---|
| 输入模型 | `model/artifacts/baseline_cnn_seed43_bn_fused_fp32.pt` |
| 结构 | Conv+ReLU+MP ×3 → GAP → Linear（无 BN，纯 Conv 带 bias） |
| FP32 测试准确率 | **0.984800** |
| 校准集来源 | MNIST **训练集**（60k，非测试集） |
| 校准集规模 | 固定 **2048** 张，seed = 12345 |
| 采样方式 | `torch.randperm(60000, generator=manual_seed(12345))[:2048]` |
| 校准索引 | 保存在 `calibration_stats.json` / `candidate_quant_config.json` 的 `indices` |
| 预处理 | 与训练一致：`ToTensor → Normalize(0.1307, 0.3081)` |

校准集索引首尾示例：`[30690, 49102, 47981, 51624, 17952, ...]`。

## 2. 校准激活分布（8 个位置，2048 张，每张含 min/max/max_abs/mean/std/p99/p99.9/p99.99/numel）

| 位置 | min | max | max_abs | p99 | p99.9 | p99.99 |
|---|---|---|---|---|---|---|
| input | -0.4242 | 2.8215 | 2.8215 | 2.8088 | 2.8215 | 2.8215 |
| stem_relu | 0.0 | 5.5023 | 5.5023 | 2.7801 | 3.8425 | 4.3651 |
| pool1 | 0.0 | 5.5023 | 5.5023 | 3.3712 | 4.1490 | 4.6298 |
| conv2_relu | 0.0 | 5.0044 | 5.0044 | 1.7970 | 2.7321 | 3.4917 |
| pool2 | 0.0 | 5.0044 | 5.0044 | 2.2712 | 3.1938 | 3.8070 |
| conv3_relu | 0.0 | **16.5255** | 16.5255 | 6.5274 | 8.9264 | 11.0336 |
| pool (GAP) | 0.0 | 4.3741 | 4.3741 | 2.9588 | 3.5720 | 4.1244 |
| logits | -18.7945 | 15.4171 | 18.7945 | 10.0359 | 12.6242 | 14.7861 |

**关键观察**：`conv3_relu` 存在重尾（max 16.53 vs p99.9 8.93），个别极端激活将满范围 scale 拉宽至 0.0648。但全范围方案仍达 98.46%，说明该尾部对精度影响可控；若未来需要更高精度，可在此位置引入 p99.9 截断（本阶段因降幅 <0.2pp 未触发）。

## 3. 权重统计（融合后 3 个卷积 + 全连接）

| 层 | shape | min | max | max_abs | per-channel max_abs 范围 |
|---|---|---|---|---|---|
| stem_conv | [16,1,3,3] | -0.8310 | 0.9230 | 0.9230 | [0.2579, 0.9230] |
| conv2 | [32,16,3,3] | -0.4610 | 0.4151 | 0.4610 | [0.1600, 0.4610] |
| conv3 | [32,32,3,3] | -1.6931 | 1.9818 | 1.9818 | [0.8585, 1.9818] |
| fc | [10,32] | -1.0198 | 0.9013 | 1.0198 | [0.8328, 1.0198] |

### 方案 A 权重 scale（per-tensor）

| 层 | scale (max_abs/127) | 饱和元素 | 饱和比例 |
|---|---|---|---|
| stem_conv | 7.268e-03 | 1 | 0.694% |
| conv2 | 3.630e-03 | 1 | 0.022% |
| conv3 | 1.560e-02 | 1 | 0.011% |
| fc | 8.030e-03 | 1 | 0.313% |

（per-tensor 下每个 layer 恰有 1 个峰值元素量化到 ±127，即 scale 定义点。）

### 激活 scale（全范围，per-tensor）

| 位置 | scale |
|---|---|
| input（signed） | 2.222e-02（max_abs/127） |
| stem_relu（UINT8） | 2.158e-02（max/255） |
| pool1 | 2.158e-02（继承 stem_relu，MaxPool 不重量化） |
| conv2_relu（UINT8） | 1.963e-02 |
| pool2 | 1.963e-02（继承 conv2_relu） |
| conv3_relu（UINT8） | 6.481e-02 |
| pool（UINT8） | 1.715e-02 |

## 4. 两种方案的量化定义

统一数值规则（本阶段所有仿真显式规定，见 `model/onn_model/int8_ptq.py`）：

```text
量化    : q = round(x / scale)
舍入    : round-half-away-from-zero（0.5→1，-0.5→-1；不用 torch.round 的 half-even）
溢出    : saturate（clamp），禁止整数回绕
权重    : signed INT8，对称，范围 [-127, 127]，zero_point = 0（不用 -128）
输入激活: signed INT8，范围 [-128, 127]，zero_point = 0
ReLU 后 : UINT8，范围 [0, 255]，zero_point = 0
累加器  : INT32（仿真用 float64 整数运算保证精确，校验落在 INT32 内并饱和）
偏置    : 量化进累加器 q_bias = round(b / (weight_scale × act_scale))，per-output-channel
反量化  : y = acc_int × weight_scale × act_scale（fake quant：FP32→整数→反量化 FP32→继续）
```

| | 方案 A（硬件简单） | 方案 B（精度优先） |
|---|---|---|
| 卷积权重 | INT8 per-tensor | INT8 **per-output-channel** |
| 全连接权重 | per-tensor | 同时测 per-tensor 与 per-output-channel |
| 激活 | per-tensor（同 A） | per-tensor（同 A） |
| 输入 | signed INT8 | signed INT8 |
| ReLU 后 | UINT8 | UINT8 |
| 累加器 | INT32 | INT32 |

MaxPool 在整数域直接对 UINT8 取最大（不重量化，scale 继承上游 ReLU）；GAP 对反量化后的实数求平均后重量化为 UINT8；最终 logits 保持 FP32（不量化输出）。

## 5. 完整测试集准确率对比（10,000 张 MNIST 测试集）

| 方案 | 准确率 | 下降 (pp) | 预测一致率 | 与 FP32 不同样本数 | logits 最大/平均误差 | NaN/Inf |
|---|---|---|---|---|---|---|
| FP32（融合） | **0.984800** | — | — | — | — | 无 |
| 方案 A（全 per-tensor） | **0.984600** | **+0.020** | 99.73% | 27 | 0.870 / 0.179 | 无 |
| 方案 B（fc per-tensor） | 0.984200 | +0.060 | 99.76% | 24 | 0.729 / 0.227 | 无 |
| 方案 B（fc per-channel） | 0.984400 | +0.040 | 99.74% | 26 | 0.761 / 0.235 | 无 |
| 方案 B + p99.9 clipping | —（未触发） | — | — | — | — | — |

> 三个方案的差异仅 0.04pp、24~27 个样本，属于量化噪声级别的差异；方案 A 最简且最优。所有方案降幅均 ≤0.2pp，因此按规范**不额外运行** p99.9 clipping（触发条件为降幅 >0.2pp）。

预测变化样本（前 20 个，含索引/真实标签/FP32 预测/INT8 预测/两组 logits）已保存在 `ptq_results.json` 的 `schemes.*.mismatches`。

## 6. 每层饱和情况（方案 A，测试集）

权重饱和（量化到 ±127 边界的元素）：每层 1 个（即 scale 定义点），比例 0.01%~0.69%，**可解释且不构成信息损失**。

激活饱和（按 low/high 拆分，避免把稀疏性误报为截断）：

| 位置 | low 比例（q==0 死神经元 / input q==-128） | high 比例（q==255 / input q==127，真正截断） |
|---|---|---|
| input | 0.00% | 0.72% |
| stem_relu | 63.32% | ≈0.00% |
| pool1 | 52.35% | ≈0.00% |
| conv2_relu | 55.90% | ≈0.00% |
| pool2 | 34.28% | ≈0.00% |
| conv3_relu | 41.95% | ≈0.00% |
| pool (GAP) | 0.02% | 0.003% |

**解释**：high（顶部截断）在所有位置≈0，说明全范围 scale 未因截断丢失信息；low 的高比例是 ReLU 死神经元稀疏性（良性）——UINT8 激活的 0 值占比 34%~63% 属正常稀疏。input 的 0.72% 是像素落在归一化范围的 ±127 边界。

INT32 累加器：各层最大累加值 49,552 / 90,257 / 54,445 / 144,498，溢出计数 0——相对 INT32 极限 ±2.1e9 有 4 个数量级余量，**无溢出风险**。

## 7. 推荐方案及理由

**推荐方案 A**（`scheme_a`）：

1. 满足全部验收标准：准确率下降 0.020pp ≤ 0.2pp ✅、无 NaN/Inf ✅、饱和可解释 ✅、重复运行结果一致 ✅（两次完整运行 JSON 逐字节相同）。
2. **最简单**：所有权重 per-tensor，硬件仅需每层一个 weight scale，无需 per-channel 存储与乘加；激活 per-tensor 与 A/B 相同。
3. 实测精度优于两个方案 B 变体（0.9846 > 0.9842 / 0.9844）。

方案 B（per-output-channel）在本模型上**无精度优势**（差距 0.02~0.04pp，噪声级），仅当后续整数参考模型暴露 per-tensor 的数值/对齐问题时才值得回退考虑。

## 8. candidate_quant_config.json

```text
experiments/model_deployment/baseline_cnn_int8_ptq/candidate_quant_config.json
```

包含：每层权重量化方式（全 per-tensor）、每层 weight scale（stem_conv 7.268e-03 / conv2 3.630e-03 / conv3 1.560e-02 / fc 8.030e-03）、每层激活 scale、输入/输出整数范围、舍入与饱和规则、校准索引（2048 个）。本阶段按要求**未导出** `.mem` / `.hex` / Verilog 参数文件。

## 9. 现有 quantization.py 是否需要修复

**需要最小修复，已修复；其余新增逻辑放在新模块 `int8_ptq.py`，未改动原文件既有 PTQ 接口（避免破坏 M2 相关测试）。**

审查结论：

| 函数 | 问题 | 处理 |
|---|---|---|
| `quantize_weights_per_channel` | ① `clamp(-128,127)` 违反权重范围 [-127,127]；② `torch.round()` 为 half-even，违反 round-half-away-from-zero | **已修复**（`model/onn_model/quantization.py`：钳位改 [-127,127]，舍入改 half-away-from-zero） |
| `calibrate_activation_range` | 只记录 min/max，无 p99/p99.9/p99.99、无 std/mean | 未破坏、可保留；本阶段由 `int8_ptq.calibrate_activations` 取代（更完整） |
| `evaluate_ptq` | **前向只反量化权重、不量化激活**——不是真正的 W8A8 仿真 | 本阶段由 `int8_ptq.evaluate_w8a8`（整数累加器 + 激活量化链）取代；未删除，M2 模型与旧测试仍可用 |

BN 融合部分（上一阶段已审计）：公式与替换逻辑正确，无需修复。

## 10. 下一阶段建立 bit-accurate 整数参考模型前仍缺少什么

1. **定点整数参考模型**：当前仿真仍是 fake-quant（float64 累加、反量化到 FP32 后继续）。bit-accurate 参考需要纯整数 MAC、累加器到激活格式的定点舍入、scale 乘法用定点移位/乘法的精确语义——尚未编写。
2. **scale 的定点表示**：当前 scale 为浮点（如 2.222e-02）。硬件友好的 scale 通常取 power-of-2 或用定点乘数；尚未定义或评估（含对精度的影响）。
3. **FPGA 友好 scale 选择**：全范围 scale 是否可四舍五入为 2 的幂而不损精度——未测。
4. **per-channel 备选路径**：虽推荐 per-tensor，但整数参考模型一旦暴露对齐问题，per-channel 的定点实现路径尚未准备。
5. **INT32 累加顺序**：当前验证了累加值在 INT32 内，但硬件累加树的特定求和顺序未固定（本模型数值小，顺序不敏感，需在参考模型中显式定义）。
6. **参数/权重定点导出格式**：权重、bias、scale 的定点二进制导出（.mem/.hex/Verilog 参数）——按本阶段要求未做。
7. **RTL / 硬件仿真器**：bit-accurate 验证需要可对照的 RTL 或参考仿真器，尚不存在。
8. **校准集与 scale 的迁移策略**：2048 张校准集已固定并保存索引，但未见 scale 对校准集规模/内容敏感性的鲁棒性检查。

## 复现

```bash
# 运行完整 PTQ 评估（默认输出到 experiments/model_deployment/baseline_cnn_int8_ptq/）
python model/evaluate_baseline_int8_ptq.py

# 自动化测试（含 INT8 PTQ 原语 20 项）
python -m pytest model/tests -q
```

输出：`calibration_stats.json`（激活/权重统计 + 校准索引）、`ptq_results.json`（三方案全量结果 + 前 20 个预测变化样本）、`candidate_quant_config.json`（推荐方案配置）。
