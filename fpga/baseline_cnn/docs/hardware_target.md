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

### 尚待冻结（To be frozen）

- 板载主时钟频率和引脚
- 复位引脚和有效电平
- UART RX/TX 引脚
- 调试 LED 引脚
- 最终顶层端口

> ⚠ 未获得可靠板卡引脚表前，**不得编造引脚约束**。

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

下一步：进入共享卷积引擎（三卷积分时复用）前，仍需以下板级资料（见 §2
尚待冻结）：板载主时钟频率与引脚、复位引脚与有效电平、UART RX/TX 引脚、
调试 LED 引脚。
