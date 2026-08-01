# BaselineCNN RTL 微架构（stem 单 MAC 串行卷积引擎）

> 本文档冻结第一版 stem 卷积引擎（`rtl/stem_conv_serial.v`）的设计。
> 数值协议见 `docs/data_format.md`；硬件目标见 `docs/hardware_target.md`；
> 参数包见 `params/manifest.json`。
>
> **定位**：这是**第一版正确性实现**，只追求与冻结的 Int8Reference 逐位一致，
> **不代表最终性能最优设计**。后续若做共享卷积引擎 / 多 lane / 行缓存，
> 本文件将成为对照基线。

## 1. 冻结的设计决策

| 项 | 冻结值 | 说明 |
|---|---|---|
| 计算并行度 | **1 个 MAC lane** | 每周期至多 1 个 `input×weight` 乘累加 |
| 输出遍历顺序 | **oc → y → x** | oc 最外、x 最内；保证输出地址单调递增 |
| 卷积遍历顺序 | **ic → ky → kx** | 本层 `Cin=1`，实际等价于 ky → kx |
| stem 层 Cin | **1** | `weight_addr` 简化但保留完整 OIHW 公式注释 |
| 卷积核 | 3×3 | stride=1，padding=1 |
| 每输出 tap 数 | 固定 **9** | 越界 tap 贡献 0，**不产生非法 RAM 地址** |
| bias | MAC 全部结束后**一次性**加入 | 符号扩展后加入，然后 INT32 饱和 |
| 精确累加 | **signed 64 bit** | 9×127² + bias 远小于 2⁶³ |
| INT32 饱和 | bias 加入后**必须**执行 | 不允许回绕（本模型实测不溢出，仍保留饱和） |
| requant | 复用已验证的 `rtl/requantize_u8.v` | **不重新实现** requant 逻辑 |
| 输出布局 | **CHW** | `(oc*28+y)*28+x`，与 golden 导出完全一致 |

## 2. 数据通路

```text
input_q  1×28×28 (S8)  ──┐
                          ▼
                     ┌────────────┐
stem_weight 16×1×3×3  │ 1× MAC lane │  acc64 (S64)
(S8, OIHW)           └─────┬──────┘
                     + bias[oc] (S32, 符号扩展)
                          ▼
                     saturate 到 S32  ── conv1_acc (调试流)
                          ▼
                     requantize_u8(MULT=0x7A999012, SHIFT=38)
                          ▼
                     stem_q (U8, CHW) ── 输出 RAM 12544
```

存储实例：

| 存储 | 模板 | 深度 | 位宽 | 内容来源 |
|---|---|---|---|---|
| input RAM | `sync_ram_u8` | 784 | S8 位模式 | PC/UART 未来经写端口写入；引擎读端口供卷积 |
| weight ROM | `sync_rom_s8` | 144 | S8 | `params/weights/stem_weight.mem` |
| bias ROM | `sync_rom_s32` | 16 | S32 | `params/biases/stem_bias.mem` |
| output RAM | `sync_ram_u8` | 12544 | U8 | 引擎写端口写入，外部读端口回读 |

## 3. 固定地址公式

```text
input_addr   = y*28 + x                      (ic 唯一，10 bit)
weight_addr  = ((oc*1 + 0)*3 + ky)*3 + kx    (完整 OIHW；ic=0 时 = oc*9 + ky*3 + kx，8 bit)
output_addr  = (oc*28 + y)*28 + x            (CHW，14 bit)
bias_addr    = oc                            (4 bit)
```

## 4. 卷积计算语义（每个输出元素）

```text
acc64 = 0
for ky = 0..2
  for kx = 0..2
    iy = y + ky - 1
    ix = x + kx - 1
    if iy / ix 越界：product = 0          # 越界 tap 贡献 0
    else:        product = signed(input_q[iy][ix]) * signed(weight[oc][0][ky][kx])
    acc64 += product
acc64 += sign_extend(bias[oc])
acc32 = saturate_int32(acc64)
stem_q = requantize_u8(acc32, 0x7A999012, 38)
```

- input 与 weight 均按 **signed 8 bit** 解释；乘积至少 signed 16 bit；
- acc64 精确累加，**不依赖算术移位**；INT32 饱和在 bias 加入**之后**；
- requant 直接例化 `requantize_u8.v`，行为与 `test_rtl_vector_contract.py` 完全一致。

## 5. 状态机

```text
IDLE ──start──▶ PROLOGUE ──▶ ACC ──(tap==8)──▶ ADD_BIAS ──▶ REQ ──(last?)──▶ DONE ──▶ IDLE
                    ▲                                                            │
                    └──────────── (next output)  ────────────────────────────────┘
```

| 状态 | 动作 |
|---|---|
| IDLE | 采样 `start`；置坐标 0；清 acc64；**busy 期间忽略新 start** |
| PROLOGUE | 清 acc64；发出 tap0 读地址（input RAM / weight ROM 在此拍采样） |
| ACC | 累加上一拍读到的 tap 数据；同时发出下一个 tap 的读地址（软件流水）；tap==8 时退出 |
| ADD_BIAS | `acc64 += sign_extend(bias[oc])`（bias ROM 上一拍采样） |
| REQ | 组合：`acc32 = saturate(acc64)` → `q = requantize(acc32)`；写输出 RAM；脉冲 `acc_valid`/`q_valid`；推进坐标 |
| DONE | `done` 单拍脉冲，`busy=0` |

- 同步 RAM 读有**一拍延迟**：发出地址的下一拍数据才有效，FSM 在 ACC 状态消费，
  **绝不在发出地址的同一拍假设数据已有效**；
- `acc_addr` / `q_addr` 从 0 严格单调递增到 12543（oc→y→x 遍历保证）；
- `done` 单拍脉冲；运行期 `busy=1`。

## 6. 周期预算

- 每输出元素：PROLOGUE 1 + ACC 9 + ADD_BIAS 1 + REQ 1 = **12 周期**；
- 全图 12544 输出 ≈ 150,528 周期 + 首尾开销；
- 单 MAC lane、单输出串行，此为正确性优先的第一版，不追求吞吐。

## 7. 同步存储约定

- 所有 RAM/ROM 均为**同步读**（`rdata <= mem[addr]` 于 posedge），一拍延迟；
- 不依赖组合读取大型数组；
- ROM 用 `$readmemh` 初始化，文件路径经**参数**传入（相对路径，无本机绝对路径）；
- Quartus 编译尽量推断为 M9K；若通用模板无法推断，记录原因，不擅自改用厂商 IP。

## 8. 与后续设计的差异（预期）

| 项 | 本版（正确性优先） | 后续方向（非本阶段） |
|---|---|---|
| 并行度 | 1 lane | 多 lane / 行缓存 |
| 层间复用 | 仅 stem | 共享 MAC 引擎分时复用三层 |
| 数据搬运 | 输入逐点随机读 | 行缓存 / 双缓冲 |
| 吞吐 | 12 cyc/输出 | 优化到 ~1 cyc/输出 |

## 9. 流式 MaxPool（maxpool2x2_stream，2026-08-01）

独立、可综合的 `2×2 stride=2 MaxPool` 流式原始模块 `rtl/maxpool2x2_stream.v`，
直接消费 stem 引擎的 `q_valid` 流：

```text
stem_q 16×28×28（UINT8，CHW，oc→y→x）
→ 2×2 MaxPool，stride 2
→ pool1_q 16×14×14（UINT8，CHW，oc→pool_y→pool_x）
```

### 9.1 存储路径的现状与最终方案

| 项 | 现状（本阶段） | 最终集成 |
|---|---|---|
| stem 完整 output RAM（12544×8） | **保留**，仅用于第一阶段逐位验证 | **删除** |
| stem → MaxPool 数据通路 | 独立模块各自验证 | stem 的 `q_valid` 流**直接**进入 MaxPool |
| 保存的特征图 | stem_q（完整 16×28×28） | **仅 pool1_q**（16×14×14） |

> 下一阶段重构 stem 存储路径（去掉 output RAM、将 `q_valid` 接到 MaxPool）
> 的前提是独立 MaxPool 已通过全部验证——本阶段已完成该验证。

### 9.2 冻结的接口与契约

```verilog
module maxpool2x2_stream (
    input  wire       clk, rst_n,
    input  wire       start,      // 单周期脉冲；busy 期间忽略
    input  wire       in_valid,   // 合格输入（posedge 采样）
    input  wire [7:0] in_q,       // UINT8 特征图值
    output reg        busy,       // 消费输入流期间为 1
    output reg        out_valid,  // 每个池化结果一个单周期脉冲
    output reg [11:0] out_addr,   // CHW 池地址，严格 0..3135，每结果 +1
    output reg [7:0]  out_q,      // UINT8 池化结果
    output reg        done        // 最后一个结果输出后的单周期脉冲
);
```

- 一次运行固定接收 **12544** 个输入，顺序 `oc=0..15 → y=0..27 → x=0..27`；
- 模块内部 `oc/y/x` 计数器**只在 `busy && in_valid` 时前进**——输入流中任意
  长度的空拍都不会推进任何状态（坐标、row_buffer、current_left、输出地址）；
- 输出顺序 `oc=0..15 → pool_y=0..13 → pool_x=0..13`；`out_addr` 严格从 0 递增
  到 3135；
- `done` 单周期脉冲，且**只在最后一个池化结果已经有效输出之后**产生；
- **禁止**在本模块中使用除法/取模/通用乘法——坐标与输出地址全部由小计数器维护。

### 9.3 流式结构

只保存上一条偶数行 `row_buffer[0:27]`（每项 UINT8，共 28×8=224 bit）：

| 输入位置 | 动作 |
|---|---|
| 偶数行 `y[0]==0` | `row_buffer[x] <= in_q`，不产生输出 |
| 奇数行 `y[0]==1`、偶数 x | `current_left <= in_q`（保存奇数行左侧值），不产生输出 |
| 奇数行 `y[0]==1`、奇数 x | 2×2 窗口齐：`max(row_buffer[x-1], row_buffer[x], current_left, in_q)` → `out_valid` |

- 四个 UINT8 最大值用**比较树**（`max_top → max_bottom → max_all`），比较为
  **unsigned**，绝不把有符号比较误用于 UINT8；
- 切到新输出通道时**不要求清空** row_buffer：新通道偶数行会在任何奇数行读取前
  完整覆盖 28 个位置；
- row_buffer 只在奇数行/奇数 x 时读取（组合读已加 guard），保证首行无 X；
- `row_buffer` 是 28×8 寄存器数组（组合读、仅 224 bit），不映射 M9K，预期落入
  逻辑（详见 §9.5 资源报告）；
- 输出 `out_valid/out_q/out_addr` 为**寄存器输出**（产生沿捕获），配合 testbench
  的 `posedge + #1` 采样无竞态。

### 9.4 验证

- `tb_maxpool2x2_stream.v` 两遍 golden 测试：
  - **测试 A（连续）**：start 后连续 12544 个 `in_valid=1`；
  - **测试 B（空拍）**：每 5 个有效输入插入 2 个空拍；
  - 两遍输出与 `pool1_q.mem` **3136/3136 逐位一致**且两遍完全相同；
  - 空拍期间内部 `oc/y/x/out_cnt` 冻结（`gap_bad=0`）；
  - 专项手工重算：`oc0 pool(0,0)/(0,13)/(13,0)/(13,13)`、`oc15 pool(13,13)`
    五个窗口直接对 4 个 stem_q 取 max 核对；
  - `out_addr` 严格 0..3135、`out_valid` 总次数 3136、有效输入 12544、
    `done` 单脉冲、输出无 X/Z；
  - 失败打印前 20 项（input_count / out_idx / oc / pool_y / pool_x /
    四个输入值 / expected / actual），非零退出。
- `model/tests/test_maxpool_rtl_contract.py`：元素数、CHW 布局与池地址映射、
  四角/中心/末窗口抽查、UINT8 范围、Python 重算全量 3136 一致。

### 9.5 综合资源（maxpool_smoke，EP4CE10F17C8，2026-08-01）

（由 `run_all.ps1` 步骤 8 实测，详见最终报告。）该原始模块为**纯逻辑**：
row_buffer 为寄存器实现，**无** M9K、**无**嵌入式乘法器、**无**除法器。

### 9.6 集成接口的三处冻结语义（勘误后的准确结论）

集成（§10）涉及的三个接口点**必须以本节的冻结语义为准**，此前文档中的
相关表述（"MaxPool start 由 stem done 触发"、"busy 可作为 stem 背压"、
"pool1 约 1 块 M9K"）均已删除/更正：

1. **MaxPool 启动时机**：**同一个 `controller_start` 同时启动 stem 和
   MaxPool**。MaxPool 必须在第一项 `stem_q_valid` 到达之前进入 RUN。
   **`stem_done` 不能用于启动 MaxPool**——它发生在全部 `stem_q` 已发送之后，
   用它启动会丢掉整个流。当前接口下，`start` 和第一项 `in_valid` **不允许在
   同一个采样沿出现**；当前集成天然满足（stem 第一项输出在 start 后约 12 周期，
   远晚于 start），仿真中必须断言这一间隔。
2. **busy 语义**：`maxpool_busy` **只是状态信号，不是 backpressure**。当前
   接口没有 ready/stall，**stem 不能被 MaxPool 暂停**。不需要背压的原因：
   MaxPool 每周期可接受一个输入，而 stem 每 12 周期才产生一个输入，前者吞吐
   始终大于后者。仿真必须断言 `stem_q_valid` 时 `maxpool_busy=1`。
3. **pool1 RAM 资源估算**：`pool1_q = 3136 × 8 = 25088 bit`。按容量理论下限
   为 **3 块 M9K**（3×9216=27648 ≥ 25088）；考虑 1024×9 等实际配置，可能映射
   为 **4 块**。**最终数字以 Quartus Fitter 报告为准**，不再写"约 1 块"。

## 10. stem+MaxPool 集成流水线（stem_pool1_pipeline，2026-08-01）

### 10.1 集成拓扑

```text
input RAM (784×S8)
→ stem_conv_serial  STORE_OUTPUT_RAM=0   (无完整 stem 输出 RAM)
→ q_valid/q_value 流
→ maxpool2x2_stream                      (流式 2×2 stride-2 MaxPool)
→ pool1 RAM (3136×UINT8, CHW, oc→pool_y→pool_x)
```

集成核心模块为 `rtl/stem_pool1_pipeline.v`。**本阶段保留独立 MaxPool 原始模块
与 stem 独立回归路径**（`tb_stem_conv_serial` / `tb_stem_conv_padding` 通过
`STORE_OUTPUT_RAM=1` 默认参数继续读回完整输出 RAM），只是集成工程不再实例化
完整的 stem 输出 RAM。

### 10.2 STORE_OUTPUT_RAM 参数

`stem_conv_serial` 新增 `parameter STORE_OUTPUT_RAM = 1`（Verilog generate）：

| 值 | 行为 |
|---|---|
| `1`（默认） | 实例化完整 12544×8 输出 RAM，`output_rdata` 读回有效；原有 testbench 与 Quartus `stem_conv_smoke` 结果不变 |
| `0` | **不实例化**输出 RAM（generate，不依赖综合器自动删除仍可能被读取的 RAM）；保留 `q_valid/q_addr/q_value` 输出流；`output_rdata` 固定为 0、`output_raddr` 不使用 |

`stem_pool1_pipeline` 内部以 `.STORE_OUTPUT_RAM(0)` 实例化 stem。

### 10.3 接口与连接（冻结）

```verilog
module stem_pool1_pipeline #(
    parameter WEIGHT_MEM_FILE = ".../stem_weight.mem",
    parameter BIAS_MEM_FILE   = ".../stem_bias.mem"
) (
    input  wire        clk, rst_n,
    input  wire        input_we, input_waddr[9:0], input_wdata(S8),
    input  wire        start,
    output wire        busy, done,
    output wire        pool_valid, pool_addr[11:0], pool_value[7:0],
    input  wire        pool_raddr[11:0], output wire pool_rdata[7:0],
    // 观察口（仅仿真/调试，不影响运行）
    output wire        stem_busy, maxpool_busy, stem_done, maxpool_done,
                       stem_q_valid, stem_q_value[7:0]
);
```

连接（内部固定）：

```text
start              → stem.start
start              → maxpool.start            # 同一个 controller_start 同启
stem.q_valid       → maxpool.in_valid
stem.q_value       → maxpool.in_q
maxpool.out_valid  → pool1_ram.we
maxpool.out_addr   → pool1_ram.waddr
maxpool.out_q      → pool1_ram.wdata
```

顶层：

```text
busy = stem_busy || maxpool_busy
done = maxpool_done              # 完成条件以 maxpool_done 为准
```

### 10.4 启动 / busy / done 语义

- **启动**：单个 `start` 拍同时启动 stem 与 MaxPool。MaxPool 在 start 拍进入
  RUN，第一项 `stem_q_valid` 在其后约 12 周期才到达，故 MaxPool 必已在 RUN。
  `start` 与第一项 `in_valid` 不在同一采样沿（集成天然满足，仿真断言 ≥2 周期）。
- **busy**：状态信号，非背压。`busy = stem_busy || maxpool_busy`；
  `maxpool_busy` 在 `stem_q_valid` 期间必须恒为 1（仿真断言）。
- **done**：`done = maxpool_done`。实测 `stem_done` 与 `maxpool_done` **同周期**
  触发（stem 的 S_DONE 紧跟最后一个 S_REQ，MaxPool 的 S_DONE 紧跟最后一个
  输入消费），但完成条件以 `maxpool_done` 为准。最后一个 pool1 写入在
  `maxpool_done` 前一周期完成，故 done 后可安全读回全部 3136 项。
- **busy 期间额外 start 被忽略**（stem 与 MaxPool 的 FSM 均只在 IDLE 响应 start）。

### 10.5 周期预算（digit8，实测）

| 度量 | 周期 |
|---|---|
| start → 第一项 stem_q | 11 |
| start → 第一项 pool1_q | 360 |
| start → 最终 done | 150529 |
| busy 总周期 | 150528 |
| stem_done 与 maxpool_done | 同周期（均落在 start+150529） |

`150528 = 12544 × 12`（stem 每输出 12 周期），pool 输出由 stem 节拍驱动。

### 10.6 验证（tb_stem_pool1_pipeline，digit8 / test index 61）

两遍推理（第二遍 reset 后重跑，结果与第一遍完全相同）：

- stem `q_valid = 12544`；stem_q 与 conv1_acc 流 **12544/12544 逐位一致**，
  `acc_addr/q_addr` 严格 0..12543；
- pool `out_valid = 3136`；pool1 流 **3136/3136 逐位一致**，`pool_addr` 严格
  连续 0..3135；
- pool1 RAM 读回 **3136/3136 一致**；`done_count=1`、`stem_done_count=1`；
- 无 X/Z、无超时（watchdog）；
- 冻结断言全部通过：`stem_q_valid ⇒ maxpool_busy`（0 违例）、done 前恰好
  12544 个 stem_q 与 3136 个 pool1_q、start 与首项 in_valid 间隔 ≥2 周期、
  stem_done 与 maxpool_done 同周期、busy 期间额外 start 被忽略、两遍结果逐位
  相同。

### 10.7 综合资源（stem_pool1_smoke，EP4CE10F17C8，2026-08-01 Fitter 实测）

集成工程 `stem_pool1_smoke`（由 `run_all.ps1` 步骤 10 编译）：Flow Successful，
LE 1401、寄存器 507、9-bit 乘法器 9、**M9K 6 块（13%）**、块内存位 32,512/423,936
（8%）、无锁存器、无截断、无意外 signed 问题、物理引脚 0、虚拟引脚 84。

M9K 分项（Fitter RAM Summary，最终数字以此为准）：

| 存储 | 模板 | 容量 | M9K | 位置 |
|---|---|---|---|---|
| input RAM | `sync_ram_u8` | 784×8 = 6272 bit | **1** | M9K_X15_Y16_N0 |
| stem weight ROM | `sync_rom_s8` | 144×8 = 1152 bit | **1** | M9K_X15_Y13_N0 |
| pool1 RAM | `sync_ram_u8` | 3136×8 = 25088 bit | **4** | X27_Y13..Y16 |
| bias ROM | `sync_rom_s32` | 16×32 = 512 bit | **0（落入逻辑）** | 12 LC / 12 reg |

- **pool1 RAM 实际为 4 块 M9K**（对应 §9.6.3 的"可能映射为 4 块"分支），
  块内存位 25088 = 4×6272（每块 1024×8 配置）；
- 总内存位 32,512 = 6272 + 1152 + 25088，**无完整 stem 输出 RAM 实例**
  （其 100352 bit 不在其中）；
- `STORE_OUTPUT_RAM=0` 使 stem 的 `out_we/out_waddr/out_wdata` 未使用，综合器
  报告 3 条 Warning 10036（对象被赋值但从不读取），属预期、非错误。
