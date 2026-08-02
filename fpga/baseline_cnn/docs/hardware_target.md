# BaselineCNN 硬件目标（AC620 / EP4CE10F17C8）

> 本文档冻结 BaselineCNN 的 FPGA 硬件目标与当前板级状态。
> 数值协议见 `docs/data_format.md`；参数包见 `params/manifest.json`（两者均与板卡无关）。

## 1. 硬件目标（已冻结）

| 项 | 值 |
|---|---|
| Board | AC620（小梅哥） |
| FPGA part | EP4CE10F17C8 |
| Family | Cyclone IV E |
| Package | F17 |
| Speed grade | C8 |
| Quartus edition | Lite |
| Quartus installation | `D:\tools\altera_lite\25.1std` |

## 2. 冻结状态

### 已冻结（Frozen）

- 开发板：AC620
- FPGA 器件：EP4CE10F17C8（Cyclone IV E）
- 模型参数和数值协议：见 `docs/data_format.md`（Int8Reference，scheme A）
- 板载主时钟：**50 MHz**，FPGA 引脚 **PIN_E1**，I/O 标准 **3.3-V LVTTL**
- 调试 LED（四颗板载 LED，均 3.3-V LVTTL）：
  - **LED0 = PIN_A2**
  - **LED1 = PIN_B3**
  - **LED2 = PIN_A4**
  - **LED3 = PIN_A3**

### 尚待冻结（To be frozen）

- 复位引脚和有效电平（AC620 V2 实物**未安装**独立 RST_N 按钮）
- UART RX/TX 引脚
- S0/S1/S2 按键引脚
- 完整 CNN 板级顶层端口（首个物理顶层 `ac620_led_bringup_top` 已落地，见 §10）

> ⚠ 未获得可靠板卡引脚表前，**不得编造引脚约束**。

> AC620 V2 实物板卡事实（2026-08-02 由板背丝印确认，见 §10）：
> - 板载主时钟晶体 50 MHz，接 **PIN_E1**；LED0/1/2/3 = **A2/B3/A4/A3**，均 3.3-V LVTTL；
> - 实物只安装 **S0、S1、S2** 三个按键；PCB 虽有 `RST_N` 丝印区域，但**未安装独立 RST_N 按钮**；
> - UART RX/TX 与 S0/S1/S2 引脚本阶段**仍未可靠冻结**；
> - **不得**根据网络资料或其他 AC620 版本猜测这些引脚。

## 3. 微架构方向（当前拟采用）

- 一套共享 MAC 计算引擎
- 三个卷积层分时复用
- 第一版优先正确性和低资源占用
- 第一版 UART 以 784 字节 `input_q` 作为输入

## 4. 算术 smoke 阶段（已完成，2026-08-01）

为在真实器件上先行确认定点算术正确性与资源预算，已落地一个**组合算术 smoke 工程**
`baseline_cnn_smoke`（`quartus/`），包含 `rtl/requantize_u8.v` 与 `rtl/gap_div49.v`：

- Questa 黄金向量验证：requant 20384 项（stem/conv2/conv3）+ GAP 穷举 12496 项 + 32 通道，全部逐位一致；
- Quartus compile flow 在 **EP4CE10F17C8（Cyclone IV E）** 上成功：Flow Successful，LE 1030、寄存器 100、9-bit 乘法器 8、无锁存器、无截断/signed 错误、物理引脚 0；
- 顶层为 `baseline_cnn_smoke_top`（寄存器包装，所有端口含虚拟时钟 `clk` 均为 VIRTUAL_PIN，**不占用物理引脚**）——Quartus Fitter 拒绝纯组合端口上的 VIRTUAL_PIN（Error 171016），故加一级寄存器包装；该顶层**不是最终板级顶层**，真实 top 落地时将删除或替换。
- 统一入口：`scripts/run_all.ps1`（工具链探测 → 器件检查 → Questa → Quartus smoke → Python 契约测试）。

## 5. stem 卷积引擎阶段（已完成，2026-08-01）

在算术 smoke 之上落地**第一版单 MAC 串行 stem 卷积引擎** `rtl/stem_conv_serial.v`
（冻结设计见 `docs/rtl_microarchitecture.md`）：

- 实现 `input_q → stem Conv2d(1→16, 3×3, pad1) → +bias → saturate INT32 → requantize_u8 → stem_q`；
- Questa 验证：conv1_acc 12544/12544、stem_q 流 12544/12544、输出 RAM 读回 12544/12544
  全部逐位一致；padding 专项（四角 4 / 边缘 6 / 中心 9 有效 tap、越界 tap 不读非法地址）通过；
- Quartus smoke 工程 `stem_conv_smoke` 在 **EP4CE10F17C8** 上 Flow Successful：
  LE 697、寄存器 166、9-bit 乘法器 9、**M9K 18 块（39%）**、块内存位 107,776/423,936（25%）、
  物理引脚 0、虚拟引脚 64；
- M9K 分项（勘误后的准确结论，证据见下）：**input RAM 1 + output RAM 16 + weight ROM 1 = 18**，
  **bias ROM 未映射 M9K**（16×32=512 bit 太小，Fitter 落入逻辑）。此前"四个存储全部推断为
  M9K（1+16+1+1=19）"的说法**有误**，已修正；M9K 总数 18 与 fit 报告一致；
  - 证据：fit.rpt 编译层次节点 `u_bias_rom` 显示 Memory Bits=0、M9Ks=0、12 LC/12 寄存器；
    map/fit 的 Fitter RAM Summary 只列出 3 个 altsyncram 实例（input/output/weight）；
    "Total block memory bits 107,776 = 6272 + 100352 + 1152"（不含 bias 的 512 bit）；
    "Total block memory implementation bits 165,888 = 18 × 9216"（18 块 M9K 的配置容量）。
  - 三个 M9K 存储均由通用同步模板（`sync_ram_u8.v` / `sync_rom_s8.v`）推断（altsyncram），
    未用厂商 IP；weight ROM 由 `$readmemh` 初始化并经综合转为 MIF。
- 顶层 `stem_conv_smoke_top` 为寄存器包装 + 全部 VIRTUAL_PIN；调试流（acc/q 地址与值）在
  包装层压缩为 8-bit 计数与 XOR，避免虚拟引脚爆炸；**不是最终板级顶层**。

> 该资源仅为 stem 单 MAC 层，不是完整 CNN（conv2/conv3/MaxPool/GAP/FC 未实现）。

## 6. 流式 MaxPool 阶段（已完成，2026-08-01）

在 stem 引擎之上落地**独立流式 `2×2 stride=2 MaxPool` 原始模块**
`rtl/maxpool2x2_stream.v`（冻结设计见 `docs/rtl_microarchitecture.md` §9）：

- 直接消费 `stem_q` 流（oc→y→x，每拍一个 UINT8）→ 输出 `pool1_q` 16×14×14；
  只保存上一条偶数行（`row_buffer` 28×8=224 bit），无除法/取模/通用乘法，
  坐标与输出地址全部用计数器维护，UINT8 比较树取 max；
- Questa 两遍 golden 验证：连续输入与"每 5 发插 2 空拍"输入均 **3136/3136 逐位一致**、
  两遍输出完全相同；空拍期间内部计数冻结（gap_bad=0）；`out_addr` 严格 0..3135、
  `done` 单脉冲、输出无 X/Z；专项手工重算 5 个窗口通过；
- Quartus smoke 工程 `maxpool_smoke` 在 **EP4CE10F17C8** 上 Flow Successful，
  纯逻辑（LE/寄存器详见最终报告）：**无 M9K、无嵌入式乘法器、无除法器**，
  `row_buffer` 由寄存器实现；物理引脚 0、全部虚拟引脚；
- 顶层 `maxpool_smoke_top` 为寄存器包装 + 全部 VIRTUAL_PIN，另带
  `dbg_incnt/dbg_outcnt`（16-bit 计数）作为完成观测量；**不是最终板级顶层**。

> 本阶段**保留** stem 完整 output RAM（仅用于第一阶段验证，通过
> `STORE_OUTPUT_RAM=1` 默认参数保持历史回归路径）；真正的集成在 §7 完成——
> stem 的 `q_valid` 流直接进入 MaxPool、只保存 `pool1_q`（详见
> `docs/rtl_microarchitecture.md` §9.6 / §10）。

## 7. stem+MaxPool 集成阶段（已完成，2026-08-01）

在独立验证的 stem 引擎与流式 MaxPool 之上落地**集成核心**
`rtl/stem_pool1_pipeline.v`（冻结设计见 `docs/rtl_microarchitecture.md` §10）：

```text
input RAM → stem_conv_serial (STORE_OUTPUT_RAM=0) → 流式 MaxPool → pool1 RAM
```

- **同一 `controller_start` 同时启动 stem 与 MaxPool**；`stem_done` 不用于启动
  MaxPool（它发生在全部 stem_q 发送之后）；`start` 与第一项 `in_valid` 不在
  同一采样沿（stem 首项输出在 start 后约 11 周期，天然满足）；
- **`busy = stem_busy || maxpool_busy` 只是状态信号，不是背压**：当前接口无
  ready/stall，stem 不被 MaxPool 暂停（MaxPool 每周期可收一个输入，stem 每
  12 周期才产一个输入，无需背压）；
- **`done = maxpool_done`**；实测 stem_done 与 maxpool_done 同周期，完成以
  maxpool_done 为准；
- Questa 两遍推理（第二遍 reset 后重跑）逐位一致：stem_q/conv1_acc 流
  12544/12544、pool1 流 3136/3136、pool1 RAM 读回 3136/3136；`stem_q_valid ⇒
  maxpool_busy`、pool 地址严格连续 0..3135、done 前恰好 12544 个 stem_q 与
  3136 个 pool1_q、busy 期间额外 start 被忽略、无 X/Z、无超时；
- Quartus 集成工程 `stem_pool1_smoke` 在 **EP4CE10F17C8** 上 Flow Successful；
  M9K 分项（Fitter 实测，见最终报告）：**input RAM 1 + weight ROM 1 + pool1 RAM
  4 + bias 落入逻辑（12 LC/12 reg）**；完整 stem 输出 RAM 已从集成工程消失
  （块内存位 32,512 = 6272 + 1152 + 25088）。（此前的"pool1 约 1 块 M9K"
  估算已在 `docs/rtl_microarchitecture.md` §9.6 更正为以 Fitter 为准。）

> 集成仍只覆盖 `input→stem→pool1`，conv2/conv3/GAP/FC 未实现（见
> `docs/data_format.md` 数据协议）。本阶段保留 stem 独立回归（`STORE_OUTPUT_RAM=1`）
> 与独立 MaxPool 原始模块。

## 8. 共享 conv2/conv3 引擎阶段（已完成，2026-08-01）

在独立验证的 stem/集成之上落地**共享 conv2/conv3 单 MAC 卷积引擎**
`rtl/conv_u8_serial.v`（冻结设计见 `docs/rtl_microarchitecture.md` §11），
`layer_sel` 锁存切换两个层；流式 MaxPool 参数化以支持 pool2：

- Questa 黄金验证：**conv2+pool2**（`tb_conv2_pool2`：conv2_acc/conv2_q
  6272/6272、pool2_q 1568/1568 逐位一致，conv2 与 pool2 同一 start 同启、
  conv_done 与 pool_done 同周期）与 **conv3**（`tb_conv3_serial`：conv3_acc/
  conv3_q 1568/1568 两遍一致、reset 后重跑逐位相同）；非法 fm_raddr 0 次、
  无 X/Z、busy/done 协议断言通过；
- Quartus smoke 工程 `conv23_smoke` 在 **EP4CE10F17C8** 上 Flow Successful；
  M9K 分项（Fitter 实测）：**conv2 weight ROM 8 + conv3 weight ROM 9 +
  两个 bias ROM 共享 2 + FM RAM 4 = 23 块 M9K（50%）**；两个 bias ROM 均映射
  M9K（与 stem bias 落逻辑不同）；无锁存器、无截断、无除法器/模运算器，
  该项为 retiming 前 conv23 smoke 结果；A+ 后完整 AC620 Fitter 实测为 9-bit
  乘法器元素 19 / DSP blocks 11（见 §12）；
- 本阶段**未做完整网络集成**（不接 stem_pool1_pipeline、不实现 GAP/FC/UART），
  conv2/conv3 的输入特征图由外部 FM RAM 提供（引擎驱动 `fm_raddr`）。

下一步：将该引擎接入完整网络（input→stem→pool1→conv2→pool2→conv3→GAP→FC）
前，仍需以下板级资料（见 §2 尚待冻结）：复位引脚与有效电平、UART RX/TX 引脚、
S0/S1/S2 按键引脚（板载主时钟与调试 LED 引脚已冻结，见 §10）。

## 9. 完整纯计算核心阶段（已完成，2026-08-02）

在共享 conv2/conv3 引擎之上落地**完整 BaselineCNN 纯计算核心**
`rtl/baseline_cnn_core.v`（冻结设计见 `docs/rtl_microarchitecture.md` §14，
无 UART / 无真实引脚 / 无多 MAC）：

- **端到端流水线** `input_q → stem → pool1 → conv2 → pool2 → conv3 → GAP → FC
  → Argmax → prediction`；控制器状态机
  `IDLE→STEM_START→STEM_WAIT→CONV2_START→CONV2_WAIT→CONV3_START→CONV3_WAIT
  →FC_START→FC_WAIT→DONE`，conv2+pool2 同启、conv3+GAP 同启、done 搭档同周期；
- **存储上限**：仅 input_q / pool1_q / pool2_q RAM + gap_mem 寄存器，**不实例化**
  完整 stem_q / conv2_q / conv3_q RAM；
- **Questa 全量验证**（`tb_baseline_cnn_core`）：digit8 完整黄金 trace 11 个节点
  全部逐位一致（12544/3136/6272/1568/1568/32/10/1），prediction=8；10 个 smoke
  样本 digit0..digit9 **无 reset 连续运行** prediction 0..9 全对、每帧 fc_acc 逐位
  一致、每帧 done 恰一次；A+ P=3 后总周期 1,590,315（31.8063 ms @ 50 MHz）；
- **Quartus 完整核心工程** `baseline_cnn_core_smoke` 在 **EP4CE10F17C8** 上 Flow
  Successful：LE **3,114**、寄存器 **928**、9-bit 乘法器 **19**、**M9K 29 块
  （63%）**、块内存位 160,512（38%）、物理引脚 0、虚拟引脚 69；
  M9K 分项：input RAM 1 + stem weight ROM 1 + pool1 RAM 4 + conv2 weight ROM 8 +
  conv3 weight ROM 9 + conv2/conv3 bias ROM 各 2 + pool2 RAM 1 + fc weight ROM 1
  = 29；**fc bias ROM 落入逻辑（10 LC）**（同 stem bias 12 LC）；无锁存、无截断、
  无实际 signed 警告、无除法器/模运算器；
- 本阶段核心顶层为寄存器包装 + 全部 VIRTUAL_PIN（压缩观测：stage/计数器/XOR/
  最后地址/最后值/prediction/done），**不是最终板级顶层**。

下一步进入 AC620 上板前，仍需冻结（见 §2）：复位引脚与有效电平、UART RX/TX
引脚、S0/S1/S2 按键引脚（板载主时钟与调试 LED 引脚已冻结，见 §10）。

## 10. AC620 最小板级 bring-up 阶段（已完成，2026-08-02）

在完整纯计算核心之上落地**首个真实物理顶层** `rtl/ac620_led_bringup_top.v`
（`input clk_50m → output [3:0] led`），用真实 50 MHz 时钟与四颗板载 LED 验证
物理引脚约束、时钟约束、Quartus 编译与 SOF 下载链路：

- **引脚冻结（板背丝印确认，2026-08-02）**：主时钟 50 MHz @ PIN_E1、
  LED0=PIN_A2、LED1=PIN_B3、LED2=PIN_A4、LED3=PIN_A3，全部 **3.3-V LVTTL**；
- **AC620 V2 实物事实**：仅 S0/S1/S2 三个按键；PCB 有 `RST_N` 丝印区域但未安装
  独立 RST_N 按钮；UART RX/TX 与 S0/S1/S2 引脚本阶段**仍未可靠冻结**；**不得**
  按网络资料或其他 AC620 版本猜测引脚；
- **顶层设计**：Verilog-2001，自由运行二进制计数器（`CNT_WIDTH` 默认 25，
  LED0..LED3 分别接 `cnt[21..24]`，相邻 LED 翻转频率严格 2:1，全部人眼可见
  闪烁），**无复位端口、无 PLL、无厂商 IP、无门控时钟**；计数器声明即赋 0，
  上电初值确定（Quartus 以 power-up 值综合）；
- **Questa 验证**：`tb/tb_ac620_led_bringup.v` 以缩小参数（宽度 8、tap 4..7）
  运行两个完整周期，断言计数器持续自增、四 LED 均翻转、相邻 LED 翻转次数严格
  二倍关系、无 X/Z，输出 `AC620_LED_BRINGUP_PASS`，已并入 `run_questa.ps1`；
- **Quartus 实体工程**：`quartus/ac620_led_bringup.{qpf,qsf}`（由
  `scripts/create_ac620_led_bringup_project.tcl` 生成并提交），TOP_LEVEL_ENTITY
  = `ac620_led_bringup_top`，器件 EP4CE10F17C8、Verilog-2001、**无 VIRTUAL_PIN**；
- **时序约束**：`quartus/ac620_led_bringup.sdc`：
  `create_clock -name clk_50m -period 20.000 [get_ports {clk_50m}]`，
  QSF 以 `SDC_FILE` 引用；编译日志**不再**出现
  "Timing requirements not specified"；
- **编译与下载产物**：独立入口 `scripts/run_ac620_led_bringup.ps1`（建工程 →
  完整 compile flow → 断言 Flow Successful / 器件 / 5 物理引脚 / 无虚拟引脚 /
  SDC 生效 / 无时序未约束 → 报告 setup/hold slack 与 Fmax → 打印 SOF 路径）；
  生成的 `.sof` 位于 `fpga/baseline_cnn/quartus/ac620_led_bringup.sof`
  （供 Quartus Programmer 下载到 AC620）；
- 该实体板工程**不**加入 `run_all.ps1` 常规回归（避免拖慢 RTL 开发），仅提供
  独立入口；Questa 部分（LED TB）随 `run_questa.ps1` 正常回归。

> 本阶段不实现 UART、不将 `baseline_cnn_core` 接入板级顶层、不虚构复位引脚；
> 完整 CNN 的板级顶层（input 装载 / 复位 / UART）留待后续阶段。

## 11. 固定 digit8 CNN 板级自检（retiming 前基线，2026-08-02）

`rtl/ac620_cnn_selftest_top.v` 是当前完整 CNN 的板级自检顶层。它仅导出
`clk_50m` 与 `led[3:0]`：内部 POR 自动释放后，从
`sim/vectors/golden_trace/input_q.mem` 的同步 ROM 装载固定 digit8 的 784 字节
`input_q`，随后发出单周期 `start`，等待 `baseline_cnn_core` 的 `done`，并检查
`prediction == 8`。加载期间和运算期间 LED 同步心跳；结果阶段以同步全闪表示 PASS、
以 `1010/0101` 交替表示 FAIL，故不依赖 LED 的有效电平极性。

- Questa 板级自检已通过；证据日志为
  `fpga/baseline_cnn/sim/tb_ac620_cnn_selftest.log`，其中含
  `AC620_CNN_SELFTEST_PASS ALL_PASS`。
- retiming 前，`quartus/ac620_cnn_selftest` 面向 **EP4CE10F17C8** 的完整编译曾为
  Flow Successful；资源为 LE **2,823**、寄存器 **962**、M9K **30**、9-bit
  乘法器 **19**、PLL **0**。相对 `baseline_cnn_core`，固定输入 ROM 额外占用
  **1 个 M9K**。
- 顶层 QSF 只有时钟与四个 LED 共 5 个物理引脚，未使用 `VIRTUAL_PIN`；SDC 对
  `clk_50m` 施加 `20.000 ns` 时钟约束并调用 `derive_clock_uncertainty`。
- 该 retiming 前基线不能在 50 MHz 烧录运行：Slow 85C setup slack 为
  **-16.968 ns**，Slow 0C 为 **-13.826 ns**，Fast 0C 为 **+3.736 ns**；Slow
  85C Fmax 为 **27.05 MHz**。最差路径为
  `conv_u8_serial.acc64[42] -> gap_stream_u8.out_q_r[6]`。
- 此旧 `.sof` 仅作本地历史基线，**不得下载到 FPGA**；A+ retiming 后的重新编译
  结果见 §12。UART、按键与 EPCS Flash 在本自检中均未使用。
- 本地 `.rpt`、`.log`、`.sof` 以及 Quartus 数据库均不是 Git 交付物；本条记录也
  不宣称完整 CNN 已在真实开发板上成功运行。

## 12. A+ Requant retiming（功能、50 MHz STA 与实板 JTAG 均通过，2026-08-02）

stem 与共享 conv2/conv3 的组合 requant 已替换为厂商无关的三级
`requantize_u8_pipe`：SAT32 → signed S32×S32→S64 MUL → 固定 SHIFT 的
ROUND_SHIFT_SAT；原 `requantize_u8.v` 保留为 oracle。ADD_BIAS 边沿显式锁存
`acc64 + sign_extend(bias)`，地址/last 在引擎内保持，所有输出仅在
`S_EMIT && pipe_out_valid` 有效。

- stem/conv2/conv3 start→done 分别为 188,161 / 940,801 / 460,993；
- 完整核心固定 **1,590,315 cycles = 31.8063 ms @ 50 MHz**；
- 流水单元、conv2+pool2、conv3、conv3+GAP、stem、padding、stem+pool1、完整
  digit8 trace、10 smoke、AC620 selftest 的 Questa 局部回归均通过；
- digit0..9 prediction 全对，AC620 selftest prediction=8；`run_all.ps1` 14/14、
  完整 pytest 219 passed；
- MaxPool/GAP/FC/argmax 功能逻辑未改，三组 producer/consumer co-done 保持；
- SDC 仍为 20.000 ns，未添加 false path 或 multicycle。

AC620 完整 Quartus 验收编译及从可追溯提交执行的正式 SOF 构建均为 Flow/Fitter
Successful：

- LE **2,933**、寄存器 **1,243**、M9K **30**、memory bits **166,784**；
- 9-bit 乘法器元素 **19**、DSP blocks **11**、PLL **0**；
- 物理引脚 **5**、虚拟引脚 **0**；
- Slow 85C setup/hold **+2.274/+0.429 ns**；
- Slow 0C setup/hold **+3.853/+0.400 ns**；
- Fast 0C setup/hold **+12.380/+0.150 ns**；
- 最小 Fmax **56.41 MHz**。

retiming 前 `conv_u8_serial.acc64[42] → gap_stream_u8.out_q_r[6]` 的
**-16.968 ns** 最差 setup 路径已消失；当前最差 setup 路径为 conv3 权重 ROM 地址
寄存器到 `conv_u8_serial.acc64[63]`，仍有 **+2.274 ns** 裕量。未降低 SDC，未添加
false path 或 multicycle。因此 RTL/STA 已满足 **50 MHz 烧录资格**。

### 12.1 正式 SOF 与 AC620 V2 实板验证

retiming 已以 commit `8948e78fb12ae0e1ae2981691396545256cf437c` 提交并推送至
`origin/fpga/retime-requant-50mhz`。随后从该 tracked-clean commit 执行正式构建：

- `run_ac620_cnn_selftest.ps1` 退出码 **0**，Flow/Fitter Successful，报告守卫输出
  `AC620_CNN_SELFTEST_REPORT_GUARDS_OK`；正式构建资源与上述验收编译一致；
- 正式 SOF 路径为 `fpga/baseline_cnn/quartus/ac620_cnn_selftest.sof`，大小
  **358,717 bytes**，SHA256 为
  `EE1A5B93504EBBCFF0954D4F99D7B504E2E0FF81DE8412EBF7EB255F13E03DE8`；
- Windows 检测到 `Altera USB-Blaster` 状态 OK；实际 cable 为
  `USB-Blaster [USB-0]`，JTAG 链位置 **1**，IDCODE **`0x020F10DD`**。该 IDCODE 在
  Quartus 中显示共享候选字符串，结合已确认的 AC620 V2 板型、板载
  EP4CE10F17C8 及同板 LED bring-up 历史接受该识别；正式配置时 Programmer 进一步
  报告 `EP4CE10F17@1`；
- JTAG SRAM 配置于 2026-08-02 成功完成：Programmer 退出码 **0**、SOF checksum
  **`0x004216F2`**、IDCODE `0x020F10DD`、`1 device(s) configured`、0 errors / 0
  warnings；
- 实板结果显示 prediction=8 的原始 `1000` 阶段与 PASS 动画 `0000/1111` 交替：
  用户观察到“一颗与三颗不同”中间穿插“四颗全同”，与 RTL 协议完全一致。因此固定
  digit8 完整 CNN selftest 的实板结果为 **prediction=8、PASS**；
- 本次仅写入 FPGA SRAM，**未写入或擦除 EPCS Flash**；断电后配置消失。`.sof`、
  Quartus 报告和日志仍为本地忽略产物，正式 SOF 通过 commit、大小和 SHA256 追溯。
