# Frozen T=4 SNN PTQ 与整数参考记录

日期：2026-09-20。状态：PTQ、训练集 calibration、三 seed 完整 test inference 和整数
reference 及内部状态位宽压缩已完成；未重新训练、未做 QAT、未进入 RTL。完整机器可读
结果见 [`results/ptq_summary.json`](results/ptq_summary.json) 和
[`results/state_quantization_summary.json`](results/state_quantization_summary.json)。

## 1. Frozen model identity

本轮以工作目录的 [`frozen_model.md`](../../experiments/snn/conv_small/frozen_model.md)、实际
checkpoint 和 inference 代码为准。网络为：

```text
latency map [B,1,8,8], value -1 or 0..3
→ Conv 1→16, 3×3, pad 1, no bias → IF
→ Conv 16→32, 3×3, stride 2, pad 1, no bias → IF
→ Linear 512→10, no bias
→ four-step time-weighted current readout → argmax
```

Conv1/Conv2 输出 shape 分别为 `[B,16,8,8]` / `[B,32,4,4]`，参数量分别为
144 / 4,608；readout 参数量 5,120，总计 9,872。两个隐藏层各有膜电位状态；readout
只有逐步加权分数。隐藏 IF 的实际公式是 `integrated = membrane + current`，严格
`integrated > 1.0` 时发放，随后 subtract-one reset。它没有 leak。

提示词中的 `beta=0.5` 经代码核对是 **readout 时间权重参数**，不是 LIF membrane beta，
因此不存在 `beta × membrane`，也不得用算术右移替换膜电位更新。FP32 权重为
`[25,20,15,10]/7`，模型末尾再除以 4；整数 reference 使用等价正比例
`[5,4,3,2]`。共同正比例不改变 argmax，分类判据与冻结模型一致。

encoder 保持既有浮点器件曲线与 Quantile mapping，在输出 latency map 处划分边界；本轮
只整数化该边界之后的 SNN core。三个 checkpoint 均以记录中的完整 SHA256 校验后加载。

## 2. Weight-only INT8

Conv1、Conv2 和 Linear 均使用独立的 per-tensor symmetric INT8：`q∈[-127,127]`、
zero-point 0、`scale=max(abs(W))/127`。转换采用 round-to-nearest、half-away-from-zero，
随后显式 saturate；推理时仅 weight fake-quant/dequant，膜电位和阈值仍为 FP32。

| Variant | Seed 7 | Seed 17 | Seed 27 | Mean ± sample std | Δ vs FP32 mean |
|---|---:|---:|---:|---:|---:|
| Frozen FP32 | 93.47% | 93.60% | 93.33% | 93.47% ± 0.14 pct | 0.00 pct |
| Weight-only INT8 | 93.44% | 93.18% | 93.11% | 93.24% ± 0.17 pct | -0.22 pct |
| Integer reference | 93.44% | 93.18% | 93.11% | 93.24% ± 0.17 pct | -0.22 pct |

三 seed mean 只下降 0.22 pct，per-tensor 已足够，本轮没有引入 per-channel 复杂度。
三次完整运行的结果一致且没有训练步骤。

## 3. Calibration 与动态范围

范围只用固定 training indices `[0,1024)`，不使用 validation/test 调位宽。以下为 seed 17
FP32 checkpoint；P99/P99.9 均指绝对值分位数。

| Tensor / state | Min | Max | Abs max | Abs P99 | Abs P99.9 |
|---|---:|---:|---:|---:|---:|
| Conv1 current | -15.0017 | 2.5598 | 15.0017 | 4.5166 | 8.0101 |
| LIF1 integrated | -16.2418 | 2.5598 | 16.2418 | 9.4801 | 14.1949 |
| LIF1 post-reset | -16.2418 | 1.5598 | 16.2418 | 9.4801 | 14.1949 |
| Conv2 current | -4.2775 | 5.8986 | 5.8986 | 1.9653 | 3.1030 |
| LIF2 integrated | -5.8957 | 7.4010 | 7.4010 | 2.4810 | 3.7381 |
| LIF2 post-reset | -5.8957 | 6.4010 | 6.4010 | 1.6869 | 2.8776 |
| Readout current | -27.4866 | 9.6970 | 27.4866 | 13.3824 | 17.8977 |
| Weighted score accumulator | -131.1620 | 49.4424 | 131.1620 | 67.8569 | 95.8086 |

## 4. Integer / fixed-point scheme

整数 core 保存 INT8 weight，不执行 dequant。第二轮状态量化先以 8 guard bits 为保守基线，
然后只比较 4 和 2 guard bits。选择只使用 validation，不使用 test：

| Scheme | Mean validation | Δ vs guard 8 | Prediction mismatch | Hidden-spike mismatch | Saturation |
|---|---:|---:|---:|---:|---:|
| Guard 8 | 94.5067% | 0.0000 pct | 0 | 0 | 0 |
| Guard 4 | 94.5067% | 0.0000 pct | 0 | 0 | 0 |
| Guard 2 | 94.4200% | -0.0867 pct | 163 | 450,327 | 0 |

Guard 2 的平均精度下降虽小，但 seed 7 单独下降 0.32 pct，并产生大量 spike 行为变化；因此
不以均值掩盖单 seed 风险。最终选择 **4 guard bits**：三 seed validation 上与 guard 8 的
spike 和 prediction 逐 bit 一致。Readout INT18→INT17 的独立 validation 也保持零 prediction
mismatch、零 saturation，故最终使用 INT17。

早期有针对性的探索还显示：强制 weight scale 为 2 的幂时 test mean 为 92.81%；取消该
约束但不给状态 guard bits 时为 92.88%。两者都比正式方案差，未保留为部署接口。

所有加法和卷积积累使用 signed integer；每个 current、membrane 更新和 readout 累加后均
saturate，不允许 wrap-around。卷积与线性累加本身无舍入；weight 和正 threshold 转换统一
采用 nearest / half-away-from-zero。阈值比较严格 `>`，spike 后只减一次整数 threshold。

下面是 seed 17 的直接 RTL 接口表；另外两个 checkpoint 使用相同格式和规则，但 scale/
threshold 由各自 weight 重新导出。位宽由四步全 fan-in worst case 覆盖，而不是按 calibration
分位数截断。

| Tensor / state | Layer | Format | Signed | Scale / frac bits | Integer range | Rule |
|---|---|---|---|---|---|---|
| Input spike | encoder boundary | UINT1 | no | — | 0/1 | `latency==t` |
| Weight | Conv1 | INT8 | yes | 0.0170355406 | -127..127 | symmetric, zp=0 |
| Raw accumulator | Conv1 | INT12 | yes | weight scale | -2048..2047 | exact 9-term MAC |
| Current | Conv1 | INT16 | yes | 0.0170355406/16 | -32768..32767 | raw accumulator `<<4`, saturate |
| Membrane | LIF1 | INT18 | yes | 0.00106472129 (4 guard bits) | -131072..131071 | saturate |
| Threshold | LIF1 | UINT10 constant | no | same as membrane | 939 | zero-extend then strict `>` |
| Spike | LIF1 | UINT1 | no | — | 0/1 | subtract-threshold reset |
| Weight | Conv2 | INT8 | yes | 0.00932901769 | -127..127 | symmetric, zp=0 |
| Raw accumulator | Conv2 | INT16 | yes | weight scale | -32768..32767 | exact 144-term MAC |
| Current | Conv2 | INT20 | yes | 0.00932901769/16 | -524288..524287 | raw accumulator `<<4`, saturate |
| Membrane | LIF2 | INT22 | yes | 0.000583063606 (4 guard bits) | -2097152..2097151 | saturate |
| Threshold | LIF2 | UINT11 constant | no | same as membrane | 1715 | zero-extend then strict `>` |
| Spike | LIF2 | UINT1 | no | — | 0/1 | subtract-threshold reset |
| Weight | Linear | INT8 | yes | 0.00879745127 | -127..127 | symmetric, zp=0 |
| Current | Linear | INT17 | yes | 0.00879745127 | -65536..65535 | exact 512-term dot product, saturate |
| Weighted score | readout | INT21 | yes | effective 0.00157097344 | -1048576..1048575 | coefficients `[5,4,3,2]`, saturate |

## 5. Verification and limitations

最终 guard-4 test 为 seed 7/17/27：93.44% / 93.18% / 93.11%，mean
**93.24% ± 0.17 pct**；与 guard-8 和 weight-only INT8 逐项一致，相对 FP32 mean 下降
0.22 pct。三个 checkpoint 的固定 1,024 张 training calibration 均为零 saturation。

自动化覆盖 deterministic weight quantization、正负 half rounding、saturation 边界、严格
threshold crossing、subtract reset、beta/readout 语义、单 timestep IF calculation、最小样本
逐层 trace、guard-bit width contract、instrumentation 和完整 repeatable inference。完整 MNIST
test inference 对三个 checkpoint 的最终选定格式均已运行。

已知限制：encoder 仍为浮点软件边界；本轮没有选择唯一 production/RTL checkpoint；scale
是每 checkpoint 的部署常量而不是 power-of-two；没有 QAT、RTL、综合、资源/时序或板级
证据。下一步应先明确供硬件实现的单一 checkpoint，再冻结 parameter export 格式并开始
Python reference ↔ RTL 的逐 timestep/逐层对拍。
