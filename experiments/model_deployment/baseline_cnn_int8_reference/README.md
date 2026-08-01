# BaselineCNN 纯整数参考模型验证报告

> 阶段：方案 A 的纯整数推理参考模型 —— 无 RTL、无 UART、无 `.mem`/`.hex`/Verilog 导出
> 日期：2026-08-01
> 输入：`model/artifacts/baseline_cnn_seed43_bn_fused_fp32.pt` + `experiments/model_deployment/baseline_cnn_int8_ptq/candidate_quant_config.json`
> 参考：FP32 = 98.48%，方案 A fake-quant = 98.46%
> 范围：仅 BaselineCNN 方案 A；不重新训练、不修改 scale、不处理 TinyResNet / 紧凑模型

**核心结论：** 纯整数参考模型的推理链路与 fake-quant 在**全部 9 个前向整数节点（含 3 个卷积累加器）上逐位一致**；第一个差异出现在 **GAP（`gap_q`）**，这是规范强制的"整数÷49"（conv3 单位）与 fake-quant"去量化均值后按 s_pool 重新量化"之间的**结构性设计差异**，而非数值错误。整数模型准确率 **98.47%**（比 fake-quant 高 0.01pp、比 FP32 仅低 0.01pp），但与 fake-quant 的预测一致率为 99.91%（9 个样本差异），**未达到 ≥99.95% 的验收线**。按规范**不放宽阈值**，如实定位差异并报告。

## 1. 输入与冻结配置

| 项 | 值 |
|---|---|
| 融合模型 | `model/artifacts/baseline_cnn_seed43_bn_fused_fp32.pt` |
| 量化配置 | `candidate_quant_config.json`（方案 A，权重全 per-tensor） |
| 权重 | signed INT8 对称，范围 `[-127, 127]`，zero_point=0 |
| 输入 | signed INT8，`[-128, 127]`，zero_point=0 |
| ReLU 后激活 | UINT8，`[0, 255]`，zero_point=0 |
| bias | INT32，`bias_scale = input_scale × weight_scale` |
| 累加器 | INT32（MAC 用 INT64 宿主 + 溢出检测 + 饱和，绝不回绕） |
| 舍入 | round-half-away-from-zero |
| 饱和 | clamp，禁止回绕 |

## 2. 纯整数链路与整数节点

```text
normalized input
→ INT8 input          (input_q)
→ INT8×INT8 卷积 + INT32 bias/accumulator   (conv1_acc)
→ 定点 requant        → UINT8 ReLU          (stem_q)
→ 整数 MaxPool                               (pool1_q)
→ conv2: conv1_acc 路径重复                  (conv2_acc, conv2_q, pool2_q)
→ conv3: conv3_acc, conv3_q
→ 整数 Global Average Pool（sum/49，round-half-away-from-zero）(gap_q)
→ INT8×UINT8 全连接   → INT32 logits         (fc_acc)
→ argmax(fc_acc)      → prediction
```

核心链路**不反量化回浮点**。浮点只用于：①启动读 scale；②一次性推导 multiplier/shift；③最后把 INT32 logits 反量化用于误差报告。

节点 dtype / 范围契约（全部验证通过）：

| 节点 | dtype | 范围 |
|---|---|---|
| input_q | int8 | [-128, 127] |
| conv1_acc / conv2_acc / conv3_acc / fc_acc | int32 | [-2³¹, 2³¹-1] |
| stem_q / pool1_q / conv2_q / pool2_q / conv3_q / gap_q | uint8 | [0, 255] |

**GAP 语义（与 fake-quant 的唯一结构性差异）：**
- 整数参考：`gap_q = round_half_away_from_zero(sum(conv3_q) / 49)`，输出保持 **conv3 单位**（s_conv3）。fc 的输入 scale 因此为 s_conv3，fc bias 按 `s_conv3 × sw_fc` 量化，最终 logits 按 `s_conv3 × sw_fc` 反量化。
- fake-quant：`q_pool = round(mean(conv3_q × s_conv3) / s_pool)`，输出为 **s_pool 单位**，fc bias 按 `s_pool × sw_fc` 量化。
- 两者都是 FP32 模型的合法逼近，但 GAP 量化网格不同 → 这是差异来源（详见 §7）。

## 3. 定点 requantization（multiplier / shift / 近似误差）

对 `real_multiplier = input_scale × weight_scale / output_scale`，近似为 `multiplier / 2^shift`：

| 层 | real_multiplier | multiplier | shift | 相对误差 |
|---|---|---|---|---|
| stem_conv | 7.4829e-03 | 2056884242 | 38 | 9.97e-11 |
| conv2 | 3.9909e-03 | 1097020857 | 38 | 3.44e-10 |
| conv3 | 4.7256e-03 | 1298974956 | 38 | 1.95e-10 |

- `multiplier` 均为正、`< 2³¹`（signed 31-bit 量级内），`shift ≥ 0`；
- `accumulator × multiplier` 用 **INT64** 中间值，右移 round-half-away-from-zero，最终饱和到 UINT8；
- 相对误差 ~1e-10 → 在全部 10k 测试集上，requant 结果与 fake-quant 的浮点舍入**逐位相同**（见 §7）；
- bias 量化：4 层 `bias_out_of_int32 = 0`（全部落在 INT32 内）。

## 4. 三模型准确率（10,000 测试集）

| 模型 | 准确率 | 相对 FP32 下降 | 与 fake-quant 之差 |
|---|---|---|---|
| FP32（BN 融合） | **0.984800** | — | — |
| fake-quant 方案 A | 0.984600 | +0.020 pp | — |
| **纯整数参考** | **0.984700** | **+0.010 pp** | **-0.010 pp** |

整数模型准确率高于 fake-quant 0.01pp，且与 FP32 的差距（0.01pp）小于 fake-quant（0.02pp）。

## 5. 预测一致率

| 对比 | 一致率 | 不一致样本数 |
|---|---|---|
| 整数 vs fake-quant | 99.910% | **9** |
| 整数 vs FP32 | 99.810% | 19 |
| fake-quant vs FP32 | 99.730% | 27 |

9 个整数-vs-fake 差异样本（前 20 个全量在 `integer_reference_results.json`）可分为两组：

- **整数对、fake 错（5 个）**：idx 582 / 1681 / 2823 / 4662 / 7928 —— 整数模型在这些样本上得到正确答案，fake-quant 错了；
- **fake 对、整数错（4 个）**：idx 490 / 1459 / 2044 / 4382。

即差异是**对称翻转**（5-4），不是整数模型系统性变差。9 个样本的 logits 最大误差 ≤ 0.158，说明是 GAP 舍入差异导致的小幅 logits 扰动越过决策边界。

## 6. 各层 accumulator 范围与 INT32 安全余量

| 层 | min | max | max_abs | 距下界余量 | 距上界余量 | 溢出 |
|---|---|---|---|---|---|---|
| stem_conv | -49,552 | 34,684 | 49,552 | 2,147,434,096 | 2,147,448,963 | 0 |
| conv2 | -90,257 | 66,958 | 90,257 | 2,147,393,391 | 2,147,416,689 | 0 |
| conv3 | -42,959 | 54,445 | 54,445 | 2,147,440,689 | 2,147,429,202 | 0 |
| fc | -39,153 | 32,354 | 39,153 | 2,147,444,495 | 2,147,451,293 | 0 |

全部 accumulator 在 INT32 内，离上下限均有 **~2.1e9** 的余量（4 个数量级），溢出计数 0，**无回绕**。

## 7. 逐层整数对比（整数参考 vs fake-quant）—— 差异定位

对每个节点，把反量化后的实数值逐元素比较（同单位节点直接比较整数）：

| 节点 | 整数一致率 | 实数值 mean/max abs err | 单位可比 |
|---|---|---|---|
| input_q | 100.0000% | 0.0 / 0.0 | 是 |
| conv1_acc | 100.0000% | 0.0 / 0.0 | 是 |
| stem_q | 100.0000% | 0.0 / 0.0 | 是 |
| pool1_q | 100.0000% | 0.0 / 0.0 | 是 |
| conv2_acc | 100.0000% | 0.0 / 0.0 | 是 |
| conv2_q | 100.0000% | 0.0 / 0.0 | 是 |
| pool2_q | 100.0000% | 0.0 / 0.0 | 是 |
| conv3_acc | 100.0000% | 0.0 / 0.0 | 是 |
| conv3_q | 100.0000% | 0.0 / 0.0 | 是 |
| **gap_q** | —（单位不同） | 1.66e-02 / 6.16e-01 | **否（首个差异）** |
| fc_acc | —（单位不同） | 4.93e-02 / 6.01e-01 | 否（gap 的传播） |

**第一个出现整数差异的位置：`gap_q`。** 整数参考在 GAP 之前与 fake-quant **逐位相同**；GAP 起，因整数÷49（conv3 单位）与 fake-quant 的 s_pool 重新量化在量化网格上不同，gap 实数值开始有 ~0.017 的平均误差，经 fc 传播后造成 9 个样本预测翻转。这不是实现缺陷，而是规范明确规定的整数 GAP 语义与 fake-quant 语义之间的固有差异。

## 8. requant 饱和（整数参考，10k 测试集）

| 节点 | low 比例（q==0 稀疏，良性） | high 比例（q==255 真截断） |
|---|---|---|
| stem_q | 63.32% | 0.0000% |
| conv2_q | 55.90% | 0.0000% |
| conv3_q | 41.95% | 0.0000% |
| gap_q | 0.12% | 0.0000% |

无真截断（high=0）；low 是 ReLU 死神经元稀疏性，与 PTQ 阶段观测一致。NaN/Inf：三种模型均无。

## 9. 验收结果（阈值未放宽）

| 验收项 | 结果 |
|---|---|
| 整数准确率与 fake-quant 相差 ≤ 0.02pp | **PASS**（0.01pp） |
| 整数 vs fake-quant 一致率 ≥ 99.95% | **FAIL**（99.91%，9 个样本） |
| 所有 accumulator 在 INT32 内 | **PASS**（overflow=0） |
| 内部整数节点 dtype / 范围符合定义 | **PASS** |
| 无整数回绕 | **PASS** |
| 重复运行逐位一致 | **PASS**（两次运行 digest 相同） |
| **整体** | **FAIL** |

按规范不放宽阈值，差异已逐层定位到 `gap_q`（§7）。唯一未达标项是"与 fake-quant 的预测一致率"，其根因是 GAP 单位的设计差异，而非整数实现的数值问题——事实上整数模型比 fake-quant 更接近 FP32（一致率 99.81% vs 99.73%）且准确率更高。

## 10. 是否满足进入参数导出阶段的条件

**严格按验收门：不满足**（第 9 节一致率一项 FAIL）。进入参数导出前需要用户决策：

1. **接受当前整数参考模型**：GAP 整数÷49（conv3 单位）是规范强制语义，模型满足全部硬件安全标准（INT32、无溢出、无回绕、确定性、dtype/range），且比 fake-quant 更接近 FP32。若验收的"对照基准"允许以 FP32 为准（整数 vs FP32 = 99.81%），则可进入参数导出。
2. **要求与 fake-quant 对齐**：把整数 GAP 改为"求和后经定点 multiplier 重新量化到 s_pool 单位"（TFLite 风格），可消除该差异，但**偏离规范强制的"整数÷49"语义**，需要用户确认这是否是期望方向。

在决策之前，本阶段不加宽阈值、不修改 GAP 语义、不进入 RTL/参数导出。

## 复现

```bash
python model/verify_baseline_int8_reference.py
python -m pytest model/tests -q
```

输出：`integer_reference_results.json`（完整指标 + 9 个差异样本 + 验收状态）。
