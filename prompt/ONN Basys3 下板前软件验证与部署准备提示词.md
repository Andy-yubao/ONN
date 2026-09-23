你现在继续开发本地仓库：

`E:\project\ONN`

当前主线以最新 `main` 分支为准。开始前先阅读当前仓库，尤其是：

- `README.md`
- `model/snn/deployment_checkpoint.md`
- `model/snn/quantization_record.md`
- `model/snn/integer_reference.py`
- `model/snn/export/`
- `fpga/README.md`
- `fpga/rtl/README.md`
- `fpga/rtl/`
- `fpga/tb/`
- `fpga/vectors/`
- `fpga/scripts/`

当前已经完成：

- Frozen T=4 SNN；
- seed 17 deployment checkpoint 冻结；
- INT8 权重与整数状态格式冻结；
- Python integer reference；
- `.mem` / `params.svh` 参数导出；
- dense RTL golden baseline；
- event-driven sparse RTL；
- Python integer reference ↔ RTL 的逐层和端到端 golden-vector 仿真对拍。

本轮任务的目标是：

> 在真正连接 Basys3 下板之前，把纯软件验证、板级通信接口、PC 端验证程序、Basys3 wrapper，以及 Vivado synthesis / implementation 准备尽可能完成，使第一次下板时新增的不确定性主要只剩真实 FPGA 执行和物理通信。

另外，本机目录：

`E:\Verilog\digital_electornics\homework\docs\basys3`

存放了我之前整理/使用过的 Basys3 资料。

请先检查其中内容。与当前项目有关的官方手册、引脚资料、约束参考等可以复制到 ONN 仓库合适的位置，例如：

`fpga/basys3/docs/`

不要修改原目录中的文件。

---

# 一、扩大 RTL 回归验证

先不要修改现有数值契约。

继续以：

`model/snn/integer_reference.py`

作为软件黄金参考。

## 1. 扩大真实 MNIST 端到端样本

当前完整 `snn_core` golden-vector 样本数量较少。

增加一套适合自动回归的真实 MNIST 样本集。

要求：

- 样本选择 deterministic；
- 明确记录 dataset split 与 index；
- 不需要为大量样本保存完整 L1/L2 巨型 trace；
- 详细逐层 trace 继续只保留少量代表样本；
- 大样本回归主要比较：
  - final weighted scores；
  - class prediction；
  - sparse synaptic-add count；
  - cycle count（若适合确定性比较）。

建议把“详细逐层验证”和“批量端到端验证”分开，避免 vectors 目录无限膨胀。

---

## 2. 增加 RTL corner cases

补充针对硬件行为的确定性测试，至少覆盖：

- 四个 timestep 全 0；
- 单个 spike；
- 角落像素 spike；
- 中心像素 spike；
- 64 bit 全 1；
- 相同位置在不同 timestep 发放；
- threshold equality；
- threshold + 1 crossing；
- 正饱和；
- 负饱和；
- frame_start 状态清零；
- 连续两帧输入；
- busy 时出现额外 `step_valid`；
- score tie 时 argmax 保留较小类别编号。

这些测试重点是 RTL 行为，不是模型 accuracy。

---

## 3. 随机 differential test

增加 Python 驱动的 deterministic randomized differential regression。

固定 random seed，生成合法：

`4 × 64-bit spike`

输入序列。

对于同一输入分别运行：

```text
Python IntegerSNNReference
vs
RTL simulation
```

至少比较：

- L1 spike；
- L2 spike；
- readout/final weighted score；
- class；
- sparse synaptic-add count。

不要生成难以审计的大型随机框架。优先保持简单、确定性和可重复。

最终应有一个统一 regression 入口，可以一次运行：

- 基础 unit tests；
- corner cases；
- 现有 golden vectors；
- 少量 detailed trace；
- 批量 MNIST；
- randomized differential tests。

---

# 二、冻结板级通信协议

在真正写 Basys3 顶层前，先设计一个简单、可调试、稳定的 PC ↔ FPGA 协议。

当前第一次板级验证只验证：

```text
4 × 64-bit spike
        ↓
FPGA SNN core
        ↓
result
```

暂时不要把真实 ADC、器件或模拟输入加入第一次板级验证。

## 推荐物理通信方式

优先使用 Basys3 板载 USB-UART。

设计简单 packet protocol。

PC 至少需要向 FPGA 发送：

- frame start；
- timestep index；
- 64-bit spike bitmap；
- 必要的 packet header / checksum 或简单完整性检测。

FPGA 至少返回：

- frame/result marker；
- predicted class；
- 10 个 final weighted scores；
- frame synaptic-add count；
- frame cycle count；
- status/error code。

协议需要：

- 明确 byte order；
- 明确 bit order；
- 明确 signed integer 编码；
- 明确 packet framing；
- 明确 reset/resync 行为；
- 支持 PC 检测 malformed packet。

请把协议写成正式 Markdown 文档，放在：

`fpga/basys3/`

附近合适的位置。

保持协议简单，不要设计成复杂通用总线。

---

# 三、编写 PC host 工具

新增 Python host-side 工具。

目标流程：

```text
MNIST image
↓
8×8 preprocessing
↓
现有 frozen encoder
↓
4 × 64-bit spike
↓
packet encoder
↓
serial transport
↓
FPGA response
↓
与 IntegerSNNReference 自动比较
```

在没有连接真实 FPGA 的情况下也必须可以测试。

因此 transport 层与逻辑层分离，例如：

```text
encoder
packet codec
transport
validator
CLI
```

增加 mock / loopback backend。

PC 端至少支持：

1. 输入指定 MNIST index；
2. 生成 4 timestep spike；
3. 计算 software golden；
4. 编码发送 packet；
5. 解析 FPGA 返回；
6. 检查：
   - 10 个 final score 是否逐 bit 一致；
   - class 是否一致；
   - counters 是否符合预期；
7. 输出清晰 PASS / FAIL。

后续还应支持批量运行多个 MNIST 样本。

不要把 pyserial 强耦合进模型代码。

---

# 四、实现 Basys3 wrapper

在现有 `onn_snn_core` 外实现板级 wrapper。

建议职责：

```text
UART RX
↓
packet parser
↓
4×64-bit timestep/frame buffer
↓
controller
↓
onn_snn_core
↓
result buffer
↓
UART TX
```

要求：

- 不修改 SNN 数值契约；
- `onn_snn_core` 尽量保持与现有 RTL 仿真兼容；
- wrapper 和 UART 属于外围接口层；
- 对 malformed packet 有确定性处理；
- 新 frame 能正确清理上一 frame 状态；
- backpressure / busy 情况行为明确。

不要为了“看起来硬件化”重写已经验证正确的 core。

---

# 五、验证 wrapper 和 UART

在进入 Vivado synthesis 前，为 wrapper 做完整仿真。

建立：

```text
PC packet
↓
UART serial waveform
↓
Basys3 wrapper
↓
SNN core
↓
UART TX waveform
↓
packet decode
```

至少验证：

- 一个简单人工输入；
- 一个真实 MNIST 样本；
- 连续多个 frame；
- malformed packet；
- reset；
- busy / packet timing 边界。

仿真最终输出仍然要与 Python integer reference 对拍。

目标是：

> 在没有真实板子的情况下，已经可以验证完整的 PC packet → RTL → result packet 逻辑链。

---

# 六、建立 Basys3 Vivado 工程

完成前述验证后，再开始建立正式 Basys3 / Artix-7 Vivado 工程。

优先使用脚本化、可重复生成方式，不要只留下一个依赖 GUI 状态的 `.xpr`。

从：

`E:\Verilog\digital_electornics\homework\docs\basys3`

获取需要的 Basys3 资料，并结合仓库中已有信息确认：

- FPGA part；
- 100 MHz clock；
- USB-UART RX/TX；
- reset/button；
- 可选 LED debug；
- 必需 XDC。

不要凭记忆猜 pin。

---

# 七、第一次 synthesis：先测当前 sparse RTL

第一次 synthesis 不要急着修改并行度。

先综合当前经过 golden-vector 验证的 sparse architecture。

记录：

- LUT；
- FF；
- BRAM；
- DSP；
- inferred RAM；
- critical path；
- WNS / TNS；
- achievable clock；
- 各主要模块资源占用。

特别检查：

- weight arrays 是否正确推断为 BRAM / ROM；
- membrane/state arrays 如何实现；
- large priority encoder；
- scatter logic；
- variable indexing；
- `%` `/` 等 RTL 写法；
- 是否存在仿真可行但综合代价异常高的结构。

生成正式 synthesis report 摘要 Markdown。

---

# 八、根据综合结果决定并行度

不要在看到 synthesis 数据之前盲目增加 parallel lanes。

根据：

- Basys3 资源；
- BRAM port 数；
- LUT；
- timing；
- 当前 cycle count；

评估：

```text
1 lane
2 lane
4 lane
```

等可能方案。

这里只需要做工程判断。

如果当前 1-lane 已满足项目需求，可以保留。

如果确实值得增加并行度，则选择最简单的一档进行实现并重新跑：

```text
RTL regression
↓
synthesis
↓
timing
```

所有优化必须保持：

```text
Integer Reference
==
RTL result
```

逐 bit 一致。

---

# 九、Implementation

当 synthesis 结构合理后执行：

- implementation；
- timing analysis；
- utilization analysis；
- DRC。

如果 timing 不通过，先定位真实 critical path，再修复。

不要为了得到绿色报告随意降低验证标准。

生成：

- utilization summary；
- timing summary；
- hardware architecture summary。

---

# 十、为下板验证准备 DEBUG_BUILD

不要把所有现有逐元素 debug 流永久连接到 UART。

增加一个简单的板级 debug 配置。

正常模式返回：

```text
class
10 final scores
synaptic-add count
cycle count
status
```

可选 DEBUG_BUILD 可以额外返回少量信息，例如：

- 每 timestep L1 spike count；
- 每 timestep L2 spike count；
- selected membrane probes；
- selected spike bitmap。

目标是出现 FPGA mismatch 时能够定位层级。

避免一次返回整个 L1/L2 tensor。

---

# 十一、建立正式板级验证脚本

虽然现在不一定连接板子，但提前写好实际下板后的验证入口。

至少支持三级测试：

### Level 1：人工输入

- all zero；
- single spike；
- corner spike；
- dense spike pattern。

### Level 2：少量 Golden MNIST

要求：

```text
FPGA final score
==
IntegerSNNReference final score
```

逐 bit 一致。

### Level 3：批量 MNIST

统计：

- total samples；
- prediction mismatch；
- score mismatch；
- communication error；
- FPGA accuracy；
- Integer Reference accuracy。

这里必须区分：

```text
FPGA accuracy
```

与：

```text
FPGA ↔ Integer Reference 一致性
```

真正验证硬件正确性的首要指标应是：

```text
prediction mismatch = 0
score mismatch = 0
```

而不是单纯看到 accuracy 接近 93%。

---

# 十二、文档同步

当前仓库中：

- `docs/project_scope.md`
- `docs/system_architecture.md`

存在明显滞后。

本轮完成后同步更新当前状态，确保不再出现：

- “尚未量化”；
- “尚无 SNN RTL”；

等已经失效的描述。

不要夸大尚未完成的板级结果。

如果当前只是 synthesis / implementation 完成，就明确写：

> 尚未完成真实 FPGA 板级验证。

---

# 十三、测试与提交原则

每完成一个明显阶段，都运行对应测试。

建议阶段顺序：

```text
1. expanded RTL regression
2. packet protocol
3. PC host + mock
4. Basys3 wrapper
5. wrapper UART simulation
6. Vivado project
7. synthesis
8. architecture adjustment if needed
9. implementation
10. board-validation tooling
11. documentation cleanup
```

不要一开始运行大量耗时任务。

需要长时间运行的 synthesis / implementation 放在对应阶段再执行。

不要删除 dense golden RTL。

不要修改 frozen checkpoint。

不要重新训练模型。

不要重新选择 champion。

不要改变 INT8 / guard-4 / threshold / temporal coefficient 数值契约，除非发现明确 bug，并先记录证据。

---

# 十四、最终向我报告

完成后请给出简洁但完整的总结：

1. 新增/修改的主要文件；
2. RTL regression 覆盖扩大到什么程度；
3. randomized differential test 结果；
4. UART packet protocol；
5. PC host 工具状态；
6. Basys3 wrapper 状态；
7. wrapper 仿真结果；
8. synthesis utilization；
9. timing；
10. 是否调整 sparse parallelism，以及原因；
11. implementation 结果；
12. 当前距离真实下板还差什么；
13. 推荐的第一次 FPGA 下板验证步骤。

如果某一步因为本机工具、Vivado 环境、缺少串口设备或其他客观条件无法完成，不要伪造结果。完成所有能够在当前机器上完成的部分，并明确记录阻塞点。

本轮工作的核心标准是：

> 下板前尽可能把软件、RTL、通信和验证链全部闭环；第一次连接真实 Basys3 时，不再重新怀疑模型数学和 RTL 数值正确性，而主要验证真实 FPGA 执行、时序和物理通信。