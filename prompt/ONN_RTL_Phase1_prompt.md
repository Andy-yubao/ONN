# ONN RTL Phase 1：基础架构、IF 与 Conv1+IF1 验证

请基于当前 `main` 分支继续开发。当前 deployment checkpoint、INT8 参数和导出文件已经冻结。

## 本阶段目标

完成 SNN RTL 的第一阶段基础实现，并建立 Python integer reference ↔ RTL 的逐 bit 验证链路。

本阶段只做到：

```text
64-bit timestep spike input
→ Conv1
→ IF1
```

不要继续实现 Conv2、IF2、FC、readout、argmax，不做 Vivado synthesis / implementation / FPGA 下板。

## 1. 建立 RTL 目录与接口

在 `fpga/` 下建立清晰目录，例如：

```text
fpga/
├── rtl/
├── tb/
├── vectors/
└── scripts/
```

RTL 必须直接使用当前冻结的：

```text
model/snn/export/
├── conv1_weight.mem
├── conv2_weight.mem
├── readout_weight.mem
├── params.svh
└── manifest.json
```

定义后续 `snn_core` 的输入协议，但本阶段无需实现完整 core。

建议核心输入语义：

```text
spike_in[63:0]   当前 timestep 的 8×8 输入 spike
step_valid       当前 timestep 有效
frame_start      新样本开始
```

## 2. 固定 RTL 数值契约

严格与现有 `IntegerSNNReference` 一致：

- weight：INT8 signed
- Conv1 current：INT16 signed
- LIF1 membrane：INT18 signed
- Conv2 current：INT20 signed
- LIF2 membrane：INT22 signed
- readout current：INT17 signed
- weighted score：INT21 signed
- LIF1 threshold：939
- LIF2 threshold：1715
- temporal coefficients：`[5,4,3,2]`
- threshold compare：严格 `>`
- reset：spike 后只减一次 threshold
- overflow：saturation，不允许 wrap-around
- signed extension、位宽截断规则必须显式处理

即使本阶段只实现 Conv1+IF1，也要将完整数值契约记录清楚。

## 3. 实现公共基础逻辑

至少完成：

- signed saturation / clamp
- IF update
- 必要的计数器、状态控制和地址生成
- Conv1 weight ROM / `.mem` 读取

IF 行为必须严格等价于：

```text
integrated = saturate(membrane + current)

if integrated > threshold:
    spike = 1
    membrane_next = saturate(integrated - threshold)
else:
    spike = 0
    membrane_next = integrated
```

## 4. 建立 Python golden-vector 生成器

复用现有：

```text
model/snn/integer_reference.py
```

生成 RTL 验证向量。

至少覆盖：

- current 为正/负
- integrated 恰好等于 threshold
- threshold ± 1
- signed saturation 上界/下界
- subtract-threshold reset
- 随机 IF 输入
- Conv1 的人工 spike pattern
- 至少若干真实 MNIST 样本在单个 timestep 下的 Conv1 current 和 IF1 输出

不要复制一套新的 Python 数值规则，golden result 必须来自现有 integer reference 或共享量化函数。

## 5. 完成 IF 单元逐 bit 对拍

建立独立 testbench。

验证：

```text
current
+ membrane
→ saturation
→ threshold comparison
→ spike
→ reset
```

RTL 输出必须和 Python golden vectors 完全一致。

## 6. 实现 Conv1 + IF1

输入：

```text
spike_in[63:0]
```

对应 8×8 UINT1 spike map。

实现：

```text
Conv 1→16
kernel 3×3
stride 1
padding 1
bias = false
INT8 weight
→ INT16 current
→ INT18 IF1 membrane
→ 16×8×8 spike
```

第一版优先：

- 正确
- 简单
- 可综合
- 易验证

不追求最大并行度。允许采用串行或有限并行 MAC。

权重顺序和地址映射必须与 `conv1_weight.mem` 导出顺序一致，并写清楚数据布局。

## 7. Conv1+IF1 对拍

至少对以下输入验证：

- 全零 spike
- 单像素 spike
- 多像素人工 pattern
- 若干真实 MNIST timestep spike

比较：

- Conv1 integer current
- IF1 integrated membrane
- IF1 post-reset membrane
- IF1 spike

要求逐元素、逐 bit 一致。

## 8. 自动化测试

尽量提供一个简单入口完成：

```text
generate golden vectors
→ run RTL simulation
→ compare
→ PASS / FAIL
```

如果当前环境已有合适 simulator，则直接使用。

如果没有，不要花大量时间安装复杂工具；保留标准 SystemVerilog testbench 和可复现运行说明即可。

## 完成标准

本阶段结束时必须满足：

- deployment 参数已真实被 RTL 使用
- IF 单元通过 bit-accurate 对拍
- Conv1+IF1 通过至少一个完整 timestep 的 bit-accurate 对拍
- 测试结果可重复
- 文档中明确当前只完成 Phase 1

最后报告：

1. 新增/修改的主要文件；
2. RTL 数据流与时序设计；
3. Conv1 的计算复用方式；
4. Python ↔ RTL 对拍覆盖范围；
5. 测试结果；
6. 尚未完成的部分。

不要开始 Conv2、IF2、FC、readout、argmax、Vivado synthesis、implementation 或 FPGA 下板。
