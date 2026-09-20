# ONN Frozen SNN PTQ 与整数参考模型任务提示词

你正在继续开发 ONN 项目。当前项目已经完成 SNN 软件侧的模型选择、三 Seed 验证、Matched CNN 对照和简单器件曲线迁移实验。现在进入 **量化与硬件参考模型阶段**。

请先完整阅读当前仓库中与最终冻结模型有关的文件和代码，再执行任务。不要凭本提示词猜测模型细节；若本提示词与仓库现状冲突，以仓库中当前冻结模型、checkpoint 索引和实际 inference 代码为准。

注意，现在的工作区应该从experiments/ 转移到/model

---

## 1. 当前已知状态

最终 Frozen SNN 已冻结为：

- 输入分辨率：`8×8`
- 时间步：`T=4`
- 编码方式：`Quantile`
- firing ratio：约 `30%`
- `beta=0.5`
- `lambda=0.10`
- 参数量：`9,872`
- 三 Seed Test：  
  - seed 7 = `93.47%`
  - seed 17 = `93.60%`
  - seed 27 = `93.33%`
  - mean = `93.47%`
  - sample std = `0.14 pct`
- Matched CNN：`97.37% ± 0.11 pct`
- 当前尚未进入量化、RTL 或 FPGA 实现阶段。

重点索引文件包括但不限于：

- `experiments/snn/conv_small/frozen_model.md`
- `experiments/snn/conv_small/records/final_three_seed_record.md`
- `experiments/snn/conv_small/records/device_curve_transfer_record.md`

请进一步定位：

- Frozen SNN 的模型定义；
- 最终 checkpoint；
- 当前 test inference 流程；
- 输出 readout / 分类判据；
- encoder 与 SNN core 的边界；
- 当前测试代码。

---

# 2. 本次任务目标

本次只做：

> **对最终 Frozen SNN 执行训练后量化（PTQ），并建立一个面向未来 FPGA/RTL 的整数 / 定点 Python reference model。**

目标不是立刻写 RTL，而是先回答：

1. 冻结模型对权重量化是否敏感？
2. 内部 membrane、threshold、accumulator 等状态需要多少位宽？
3. 在不重新训练的情况下，能够把 FP32 推理转化到多大程度的整数 / 定点运算？
4. 最终能否得到一个未来可与 RTL 逐步对拍的 bit-accurate Python reference model？

---

# 3. 重要边界

## 3.1 不等待外围电路

当前外围电路与 ADC 输入链还没有完全冻结。

这 **不妨碍本次量化工作**。

请把系统视为：

```text
analog / ADC / front-end
        ↓
input encoder
        ↓
Frozen SNN Core
```

本次重点是 **Frozen SNN Core**。

不要因为 ADC、DAC、模拟电压接口尚未确定而停止 PTQ。

对于输入端，可以：

- 继续复用当前软件 inference 中生成的 spike；
- 或在明确 encoder 边界后，从 encoder 输出开始进入量化 SNN core。

不要擅自重新设计模拟前端。

---

## 3.2 不进入 RTL / FPGA

本次不要：

- 写 Verilog / SystemVerilog；
- 创建 Vivado 工程；
- 做 synthesis / implementation；
- 做 Basys3 资源估算的大规模实现；
- 设计最终 event-driven accelerator。

允许在结果中记录对未来 RTL 有直接意义的信息，例如：

- 位宽；
- 状态数量；
- rounding；
- saturation；
- scale；
- 哪些乘法可转为 shift；
- 哪些变量必须保留更高精度。

但不要开始实现 RTL。

---

## 3.3 不做 QAT

第一阶段只做 **PTQ**。

不要重新训练模型。

不要：

- QAT；
- fine-tuning；
- 蒸馏；
- architecture search；
- 修改 Frozen SNN 网络结构；
- 为了追回精度重新训练冠军模型。

如果 PTQ 明显失败，只需记录原因和下一步建议，不要自动升级到 QAT。

---

## 3.4 控制实验复杂度

不要做大规模网格搜索。

我们之前已经完成大量 SNN 参数实验，现在的目标是硬件化，而不是重新进入模型调参阶段。

优先：

1. 最小 baseline；
2. 找到精度瓶颈；
3. 有针对性地增加 1~2 个必要方案。

不要无意义遍历几十种 bit-width / scale 组合。

---

# 4. 执行顺序

请严格按照以下顺序推进。

---

## Phase 0：审查当前 Frozen Model

先只阅读，不修改。

确认并报告：

1. Frozen SNN 的实际网络结构；
2. 每一层输入 / 输出 shape；
3. 每层参数量；
4. 哪些层带 LIF / membrane state；
5. `beta=0.5` 在实际代码中的更新公式；
6. threshold 的实际定义；
7. reset 方式；
8. 当前 output readout / 最终分类判据；
9. 当前 FP32 inference 的数据流；
10. encoder 输出到 SNN core 的具体 tensor 格式；
11. 最终 checkpoint 的加载方式。

特别注意：

> FPGA 未来的分类判据必须和当前 Frozen Model 的 inference 完全一致。

不要重新设计输出判别方式。

---

# 5. Phase 1：Weight-only INT8 PTQ Baseline

先做最简单的实验：

> **只量化权重，其他内部状态仍使用当前浮点实现。**

建议优先尝试对称 INT8：

```text
q ∈ [-127, 127]
```

基本形式可类似：

```text
scale = max(abs(W)) / 127
W_q = round(W / scale)
W_hat = W_q * scale
```

但不要机械照搬。

请根据网络中 Conv / Linear 层实际情况判断：

- per-tensor 是否已经足够；
- 如果 per-tensor 明显损失，再尝试 per-output-channel；
- 不要一开始就引入复杂量化方案。

需要明确记录：

- scale 定义；
- rounding 规则；
- clipping / saturation 规则；
- 是否保留 zero-point；
- Conv 与 Linear 是否使用同一策略。

### 必须得到的结果

至少比较：

| Model | Accuracy |
|---|---:|
| Frozen FP32 baseline | 约 93.47% |
| INT8 weight-only PTQ | 实测 |

并确认：

- inference 可复现；
- 多次运行结果一致；
- 没有偷偷重新训练。

---

# 6. Phase 2：分析动态范围

在进入完整定点化前，先进行一次 **必要但轻量的 calibration / instrumentation**。

目的不是调参，而是测量：

- 每个 LIF 层 membrane 的实际范围；
- synaptic input / accumulation 的范围；
- threshold 的量级；
- reset 前后状态范围；
- output readout 相关数值范围；
- 是否存在明显 outlier。

建议优先使用：

- 固定的一小部分训练集 / calibration set；
- 或已有验证数据中的有限样本。

不要为了统计范围跑不必要的大规模实验。

输出每层至少包括：

```text
min
max
abs max
必要时 P99 / P99.9
```

目的是决定后续 fixed-point 位宽，而不是追求统计学上的极致精确。

---

# 7. Phase 3：整数 / 定点内部状态

在掌握范围后，逐步处理：

- membrane；
- threshold；
- synaptic accumulator；
- 必要的中间状态；
- output readout 所需状态。

不要直接强制所有变量 INT8。

允许：

- 权重 INT8；
- accumulator INT16 / INT24 / INT32；
- membrane 使用适当的 signed fixed-point；
- threshold 与 membrane 共用 scale；
- 不同层使用不同 scale / bit-width。

重点是：

> **硬件合理 + 精度稳定 + 实现规则明确。**

---

# 8. beta = 0.5 的特殊处理

当前 Frozen Model：

```text
beta = 0.5
```

这是非常适合硬件实现的参数。

请检查真实模型的 membrane 更新公式。

如果数学和量化 scale 允许，应优先把：

```text
beta * membrane
```

实现成等价的：

```text
arithmetic right shift by 1
```

但要非常谨慎处理：

- signed arithmetic；
- 奇数除 2 时的 rounding；
- Python 与未来 SystemVerilog 算术右移的行为；
- 负数时 floor / truncation 的差异；
- overflow；
- saturation。

最终 Python reference 中必须明确冻结这一规则。

---

# 9. 必须显式定义的整数运算规则

为了未来 RTL 能逐 bit 对拍，本阶段不能只得到“量化后的准确率”。

必须定义：

## 9.1 Rounding

例如：

- round-to-nearest；
- ties-to-even；
- truncation；
- arithmetic shift。

必须选定一种并在代码中统一。

不要依赖 Python / PyTorch 隐式行为而不记录。

---

## 9.2 Saturation / Overflow

明确：

```text
超过最大值怎么办？
低于最小值怎么办？
```

优先考虑硬件上清晰的 saturating arithmetic，除非现有模型和实验表明 wrap-around 更合理。

---

## 9.3 Signedness

每个变量明确：

- signed / unsigned；
- bit width；
- scale / fractional bits。

---

## 9.4 Reset

必须和 Frozen FP32 模型保持语义一致。

确认：

- spike 后是 subtract threshold？
- reset to zero？
- 还是其他方式？

整数 reference 必须完全复现。

---

# 10. 建立 bit-accurate Python Reference

最终要新增一个清晰独立的 **hardware-oriented reference inference**。

它不能只是：

```text
fake quantization -> dequantize -> float PyTorch
```

最终 reference 应尽量真实地执行：

```text
integer weights
integer / fixed-point state
integer accumulator
explicit shift
explicit compare
explicit reset
explicit clipping / saturation
```

目标是未来能够做到：

```text
Python reference
      ↕
RTL simulation
```

逐层甚至逐 timestep 对拍。

---

# 11. 不要求本轮彻底整数化输入 Encoder

当前输入链涉及器件模拟、未来 ADC 和外围电路，尚未完全冻结。

因此请明确区分：

```text
Encoder / Front-end
```

和：

```text
Quantized SNN Core
```

如果当前 encoder 内仍存在浮点计算，不要为了“全整数化”而强行重构所有 device simulation。

本轮最重要的是：

> **encoder 输出 spike 之后的 Frozen SNN Core 能够转化为硬件明确的整数 / 定点模型。**

如果某些 encoder 参数未来也必须量化，请在报告中单独列为后续工作。

---

# 12. 精度实验

最终至少报告：

| Variant | Test Accuracy | Δ vs FP32 |
|---|---:|---:|
| FP32 Frozen SNN | baseline | 0 |
| INT8 weight-only | ... | ... |
| Final fixed-point / integer reference | ... | ... |

如确有必要，可增加少量中间实验，例如：

```text
INT8 weights + FP membrane
INT8 weights + 16-bit membrane
```

但不要扩展成大规模搜索表。

---

# 13. 验证要求

请增加必要的自动化测试。

至少覆盖：

1. 权重量化 determinism；
2. rounding 边界；
3. saturation 边界；
4. signed negative values；
5. `beta=0.5` shift 行为；
6. threshold crossing；
7. reset；
8. 单个 timestep 的 reference calculation；
9. 一个最小样本的逐层输出；
10. 完整 test inference 能正常运行。

如已有测试框架，复用现有结构。

不要为了这一任务重新搭建庞大的测试系统。

---

# 14. 结果记录

请保持当前项目已有的实验记录风格。

建议在当前 `experiments/snn/conv_small/` 体系下新增清晰的量化目录或记录，但请先观察仓库现有组织方式，不要擅自制造重复层级。

最终至少需要：

### 代码

- PTQ 工具；
- calibration / range instrumentation；
- integer / fixed-point reference inference；
- 必要的 test。

### 结果

- FP32 baseline；
- weight-only INT8；
- final fixed-point reference；
- 每层 bit-width / scale；
- accuracy delta。

### 文档

新增或更新一份明确的量化记录，内容至少包括：

```text
1. Frozen model identity
2. Quantization scheme
3. Weight quantization
4. State / accumulator quantization
5. Rounding rules
6. Saturation rules
7. beta=0.5 implementation
8. Accuracy comparison
9. Known limitations
10. Next step toward RTL
```

并更新必要的 README / experiment index。

---

# 15. Hardware Quantization Table

请最终生成一张表，为未来 RTL 直接使用。

例如：

| Tensor / State | Layer | Format | Signed | Scale / Frac bits | Range | Rule |
|---|---|---|---|---|---|---|
| Weight | Conv1 | INT8 | yes | ... | ... | symmetric |
| Accumulator | Conv1 | INTxx | yes | ... | ... | saturate |
| Membrane | LIF1 | Qm.n / INTxx | yes | ... | ... | ... |
| Threshold | LIF1 | same scale | yes/no | ... | ... | compare |
| Spike | LIF1 | 1 bit | no | — | 0/1 | — |

请根据实际模型填写，不要照抄示例。

这张表将成为之后 RTL 设计的重要接口文档。

---

# 16. 对 event-driven 的处理

当前模型已经观察到明显的事件稀疏性：

```text
352,256 dense-equivalent MAC/image
约 9,921 synaptic additions/image
```

但是：

> **本次不要实现 event-driven accelerator。**

只需要确保量化 reference 不破坏 spike 行为，并为未来 event-driven RTL 保留清晰的数据和整数定义。

不要据此直接宣称 FPGA 能效提升。

---

# 17. 成功标准

本任务完成时，应满足以下条件：

- [ ] Frozen FP32 baseline 能被当前代码复现；
- [ ] 已完成 weight-only INT8 PTQ baseline；
- [ ] 已记录主要内部动态范围；
- [ ] 已确定合理的 membrane / accumulator / threshold 定点方案；
- [ ] 已明确 rounding；
- [ ] 已明确 saturation / overflow；
- [ ] 已明确 signedness；
- [ ] `beta=0.5` 已采用明确、可映射到 RTL 的实现；
- [ ] 已建立 integer / fixed-point Python hardware reference；
- [ ] 已获得最终 quantized test accuracy；
- [ ] 已生成 hardware quantization table；
- [ ] 必要测试通过；
- [ ] README / 实验记录已更新；
- [ ] 没有重新训练 Frozen Model；
- [ ] 没有进入 RTL；
- [ ] 没有进行无意义的大规模 bit-width 搜索。

---

# 18. 工作方式要求

先审查，后实施。

在开始修改前，先给出一份简短的执行计划，说明：

1. 找到的 Frozen Model 实际结构；
2. 当前 inference/readout；
3. 计划如何做 weight-only baseline；
4. 计划如何测量动态范围；
5. 计划如何构造 integer reference；
6. 预计新增 / 修改哪些文件。

确认计划本身自洽后直接执行，不需要等待额外确认。

如果遇到某个细节无法从仓库确定：

- 优先从实际 checkpoint、模型和 inference 代码中推导；
- 不要凭空假设；
- 如果不影响整体任务，采用最小合理方案并记录；
- 不要因为一个非关键问题停下整个任务。

---

# 19. 最终汇报格式

任务结束后，请给出简洁报告：

```text
A. Frozen model verified
B. Weight-only INT8 result
C. Final fixed-point scheme
D. Final quantized accuracy
E. Hardware quantization table
F. Tests
G. Files changed
H. Known limitations
I. Recommended next step
```

重点报告实际结果，不要重复大段理论。

下一阶段预计是：

> **基于本次冻结的整数 reference model，设计 SNN FPGA microarchitecture，并开始 Python ↔ RTL 对拍。**
