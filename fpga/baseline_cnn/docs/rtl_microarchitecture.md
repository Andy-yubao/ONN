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
| DONE | `done` 单拍脉冲；**busy 覆盖 DONE** 保持为 1 |

- 同步 RAM 读有**一拍延迟**：发出地址的下一拍数据才有效，FSM 在 ACC 状态消费，
  **绝不在发出地址的同一拍假设数据已有效**；
- `acc_addr` / `q_addr` 从 0 严格单调递增到 12543（oc→y→x 遍历保证）；
- **busy/done 协议（冻结）**：`busy` 覆盖 PROLOGUE..DONE，从接收 `start` 起保持
  为 1，**直到 done 脉冲出现前不得提前下降**（运行开始后不存在 `busy=0 && done=0`
  窗口）；`done` 为单周期脉冲；done 高电平期间 `busy` 可为 0；外部只能在观察到
  `busy=0` 后发起新的 `start`（连续两次启动不丢失 start）。

## 6. 周期预算

- 每输出元素：PROLOGUE 1 + ACC 9 + ADD_BIAS 1 + REQ 1 = **12 周期**；
- 全图 12544 输出 = 150,528 周期（compute）+ 1（S_DONE 状态周期）= **150,529**；
  `busy` 覆盖 S_DONE；A+ P=3 后 `busy_cycles == done 周期 − start 周期 == 188161`；
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

模块已**参数化**以同时服务 pool1（stem 默认配置）与 pool2（conv2 输出配置），
默认参数保持原 stem 测试不变：

```verilog
module maxpool2x2_stream #(
    parameter N_CH       = 16,     // 特征图通道数（pool2: 32）
    parameter IN_H       = 28,     // 输入行数（pool2: 14）
    parameter IN_W       = 28,     // 输入列数（pool2: 14）
    parameter OC_W       = 4,      // oc 计数器位宽（pool2: 5）
    parameter XY_W       = 5,      // y/x 计数器位宽（pool2: 4）
    parameter OUT_ADDR_W = 12      // out_cnt/out_addr 位宽（pool2: 11）
) (
    input  wire       clk, rst_n,
    input  wire       start,      // 单周期脉冲；busy 期间忽略
    input  wire       in_valid,   // 合格输入（posedge 采样）
    input  wire [7:0] in_q,       // UINT8 特征图值
    output reg        busy,       // 从 start 起保持 1 直到 done（RUN..DONE）
    output reg        out_valid,  // 每个池化结果一个单周期脉冲
    output reg [OUT_ADDR_W-1:0] out_addr,   // CHW 池地址，每结果 +1
    output reg [7:0]  out_q,      // UINT8 池化结果
    output reg        done        // 最后一个结果输出后的单周期脉冲
);
```

- 派生几何为局部参数：`POOL_H = IN_H>>1`、`POOL_W = IN_W>>1`（移位，非除法）、
  `N_INPUTS = N_CH*IN_H*IN_W`、`N_OUTPUTS = N_CH*POOL_H*POOL_W`；
- 一次运行固定接收 **N_INPUTS** 个输入（pool1: 12544，pool2: 6272），顺序
  `oc→y→x`；
- 模块内部 `oc/y/x` 计数器**只在 `busy && in_valid` 时前进**——输入流中任意
  长度的空拍都不会推进任何状态（坐标、row_buffer、current_left、输出地址）；
- 输出顺序 `oc→pool_y→pool_x`；`out_addr` 严格从 0 递增到 N_OUTPUTS−1
  （pool1: 0..3135，pool2: 0..1567）；
- `done` 单周期脉冲，且**只在最后一个池化结果已经有效输出之后**产生；
- **busy/done 协议（冻结，同 stem）**：`busy` 覆盖 RUN..DONE，从 start 起保持
  为 1 直到 done 脉冲出现，不存在 `busy=0 && done=0` 窗口；
- **禁止**在本模块中使用除法/取模/通用乘法——坐标与输出地址全部由小计数器维护，
  半尺寸是移位。

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

- `tb_maxpool2x2_stream.v` 三遍 golden 测试：
  - **测试 A（连续）**：start 后连续 12544 个 `in_valid=1`；
  - **测试 C（无 reset 背靠背）**：A 的 done 后立即再次 start、再次喂同一连续流，
    输出与 A 逐位一致（证明连续两次启动不丢失 start）；
  - **测试 B（空拍）**：每 5 个有效输入插入 2 个空拍；
  - 三遍输出与 `pool1_q.mem` **3136/3136 逐位一致**且各遍完全相同；
  - 空拍期间内部 `oc/y/x/out_cnt` 冻结（`gap_bad=0`）；
  - 专项手工重算：`oc0 pool(0,0)/(0,13)/(13,0)/(13,13)`、`oc15 pool(13,13)`
    五个窗口直接对 4 个 stem_q 取 max 核对；
  - `out_addr` 严格 0..3135、`out_valid` 总次数 3136、有效输入 12544、
    `done` 单脉冲、输出无 X/Z；
  - **busy/done 协议断言**：运行开始后 done 前无 `busy=0 && done=0`（0 违例），
    `busy_cycles == done 周期 − start 周期`；
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
  触发（stem 的 S_DONE 紧跟最后一个 valid-qualified S_EMIT，MaxPool 的 S_DONE 紧跟最后一个
  输入消费），但完成条件以 `maxpool_done` 为准。最后一个 pool1 写入在
  `maxpool_done` 前一周期完成，故 done 后可安全读回全部 3136 项。
- **busy 期间额外 start 被忽略**（stem 与 MaxPool 的 FSM 均只在 IDLE 响应 start）。

### 10.5 周期预算（digit8，实测）

| 度量 | 周期 |
|---|---|
| start → 第一项 stem_q | 11 |
| start → 第一项 pool1_q | 360 |
| start → 最终 done | 188161 |
| busy 总周期 | 188161 |
| stem_done 与 maxpool_done | 同周期（均落在 start+188161） |

`150528 = 12544 × 12`（stem 每输出 12 周期）为 compute 周期，另 +1 为 S_DONE
状态周期；**busy 覆盖 S_DONE**，故 `busy_cycles == done 周期 − start 周期`。
pool 输出由 stem 节拍驱动。

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

## 11. 共享 conv2/conv3 单 MAC 卷积引擎（conv_u8_serial，2026-08-01）

独立的、只服务 conv2/conv3 的**共享单 MAC 串行卷积引擎**
`rtl/conv_u8_serial.v`（**不替换**已验证的 stem 引擎 `stem_conv_serial.v`）：

```text
conv2 (layer_sel=0): pool1_q 16×14×14 → Conv2d 16→32,3×3,pad1 → conv2_q 32×14×14
conv3 (layer_sel=1): pool2_q 32×7×7  → Conv2d 32→32,3×3,pad1 → conv3_q 32×7×7
```

### 11.1 层配置（冻结，与 manifest.json 一致）

| 项 | conv2 | conv3 |
|---|---|---|
| 输入（UINT8，CHW） | pool1_q 16×14×14（3136） | pool2_q 32×7×7（1568） |
| 权重（SINT8，OIHW） | 32×16×3×3 = 4608 | 32×32×3×3 = 9216 |
| bias（SINT32） | 32 | 32 |
| 输出（CHW） | conv2_q 32×14×14（6272） | conv3_q 32×7×7（1568） |
| requant multiplier | 1097020857（0x416335B9） | 1298974956（0x4D6CC8EC） |
| requant shift | 38 | 38 |

### 11.2 接口与连接（冻结）

```verilog
module conv_u8_serial #(
    parameter WT2_MEM_FILE/BIAS2_MEM_FILE/WT3_MEM_FILE/BIAS3_MEM_FILE
) (
    input  wire       clk, rst_n,
    input  wire       start,            // 单周期脉冲；busy 期间忽略
    input  wire       layer_sel,        // 0=conv2，1=conv3；start 时锁存
    output reg        busy, done,       // busy 覆盖 DONE（同 stem 协议）
    output reg [11:0] fm_raddr,         // 外部输入 FM RAM 读地址
    input  wire [7:0] fm_rdata,         // 外部同步 RAM 一拍延迟返回（UINT8）
    output reg        acc_valid, acc_addr[12:0], acc_value(S32),   // acc 调试流
    output reg        q_valid,  q_addr[12:0],  q_value[7:0]        // q 调试流
);
```

- 输入特征图 RAM **位于模块外部**：引擎驱动 `fm_raddr`，外部同步 RAM 下一拍
  提供 `fm_rdata`（与 `sync_ram_u8` 一拍延迟契约一致）；
- 模块内部实例化两个层的 weight/bias ROM 共 4 个：conv2 weight（4608）、
  conv3 weight（9216）、conv2 bias（32）、conv3 bias（32）；
- **`layer_sel` 在 `start` 时锁存到 `layer`**——运行期间外部改变 `layer_sel`
  不影响当前推理；
- 复用已验证的 `requantize_u8.v`（乘法器按 `layer` 选择），**从不重实现**。

### 11.3 遍历与地址公式（冻结）

遍历固定 `oc → y → x`（外层）→ `ic → ky → kx`（内层），输出 CHW：

```text
out_addr    = (oc*H + y)*W + x           # H=W=14 (conv2) / 7 (conv3)
in_addr     = (ic*H + iy)*W + ix         # iy=y+ky-1, ix=x+kx-1（带 padding）
weight_addr = ((oc*Cin + ic)*3 + ky)*3 + kx
```

- padding 越界 tap 贡献 0，且**不生成非法 RAM 地址**（地址总线强制为 0）；
- **不用除法/取模**：不用 tap 编号再 `/3`、`%3` 恢复坐标，直接维护 `ic/ky/kx`
  嵌套计数器（kx 到 2 → ky+1；ky 到 2 → ic+1）；
- 地址常数乘法全部写为移位/加减：`196=128+64+4`、`49=32+16+1`、
  `144=128+16`、`288=256+32`、`14=16−2`、`7=8−1`、`9=8+1`、`3=2+1`；
  不推断通用除法器/模运算器。

### 11.4 数值语义（冻结，scheme A）

```text
acc64   = Σ (UINT8 input × SINT8 weight)          # 乘积至少 17 位有符号
acc64  += sign_extend(bias)                        # 所有 MAC 之后加一次
acc32   = saturate_int32(acc64)                    # clamp，绝不回绕
q       = requantize_u8(acc32, layer_mult, 38)
```

- 输入按**无符号 0..255** 解释，权重按**有符号 −127..127** 解释；
- 精确累加使用 **signed 64-bit**；INT32 饱和在 bias 之后，不允许回绕；
- 同步 RAM/ROM 的一拍延迟在流水线中正确处理（PROLOGUE 发 tap0，MAC 累加一拍
  前发出的 tap 数据，同时发出下一个 tap 读）。

### 11.5 周期预算（实测）

| 层 | 每输出周期 | 首项 q | start → done | busy 总周期 |
|---|---|---|---|---|
| conv2 | 1+144+1+4 = **150** | 149 | **940801** | 940801 |
| conv3 | 1+288+1+4 = **294** | 293 | **460993** | 460993 |

`6272×150 + 1 = 940801`、`1568×294 + 1 = 460993`（+1 为 S_DONE 状态周期；done
脉冲在其后一周期，busy 覆盖 S_DONE 故 `busy_cycles == done − start`）。conv2 与
conv3 的固定首尾开销即这 1 个 S_DONE 周期。

### 11.6 验证（tb_conv2_pool2 / tb_conv3_serial，digit8 / test index 61）

- **conv2 + pool2**：`tb_conv2_pool2.v`。conv2 与参数化 MaxPool（N_CH=32,
  IN_H=14, IN_W=14, OC_W=5, XY_W=4, OUT_ADDR_W=11）**同一 start 同启**，非由
  conv2_done 启动 pool2：
  - conv2_acc / conv2_q 流 **6272/6272 逐位一致**，地址严格 0..6271；
  - pool2 流 **1568/1568 逐位一致**，地址严格 0..1567；
  - `q_valid ⇒ pool_busy`（0 违例）；非法 fm_raddr（>3135）**0 次**；
    `conv_done` 与 `pool_done` **同周期**、`done` 单脉冲、无 X/Z。
- **conv3**：`tb_conv3_serial.v` 两遍（第二遍 reset 后重跑）：
  - conv3_acc / conv3_q 流 **1568/1568 逐位一致**，地址严格 0..1567；
  - 非法 fm_raddr（>1567）**0 次**、`done` 单脉冲、无 X/Z；
  - **两遍结果逐位相同**（确定性）。
- `model/tests/test_conv23_rtl_contract.py`：元素数、conv2/conv3 OIHW 地址双射、
  输入/输出 CHW 双射、四角/边缘/中心 tap 数、UINT8×SINT8 解释、multiplier/shift
  与 manifest 一致、Python 全量重算 conv2/conv3 输出、全量 pool2 重算。

### 11.7 综合资源（conv23_smoke，EP4CE10F17C8，2026-08-01 Fitter 实测）

（由 `run_all.ps1` 步骤 11 编译）Flow Successful，LE 829、寄存器 141、
9-bit 乘法器 9、**M9K 23 块（50%）**、块内存位 137,728/423,936（32%）、
物理引脚 0、虚拟引脚 45。

M9K 分项（Fitter RAM Summary，最终数字以此为准）：

| 存储 | 模板 | 容量 | M9K |
|---|---|---|---|
| conv2 weight ROM | `sync_rom_s8` | 4608×8 = 36864 bit | **8** |
| conv3 weight ROM | `sync_rom_s8` | 9216×8 = 73728 bit | **9** |
| conv2 bias ROM | `sync_rom_s32` | 32×32 = 1024 bit | **2（与 conv3 bias 共享打包）** |
| conv3 bias ROM | `sync_rom_s32` | 32×32 = 1024 bit | 同上 |
| FM RAM（conv2 输入平面） | `sync_ram_u8` | 3136×8 = 25088 bit | **4** |

- 两个 bias ROM 均映射到 M9K（各 2 块、位置相同被 Fitter 打包，故总数 23 而非
  25）；**与 stem bias 落入逻辑不同**（stem bias 仅 16×32=512 bit 太小）；
- 总内存位 137,728 = 36864+73728+1024+1024+25088；
- **无锁存器、无截断**（0 条 Warning 10230）、无实际 signed 警告（20 条匹配均为
  LPM 参数显示）、**无除法器/模运算器**；
- 该表对应 retiming 前资源；A+ 后改为明确 S32×S32→S64 流水乘法，完整 AC620
  Fitter 实测为 9-bit 乘法器元素 19 / DSP blocks 11（见 §15.4）；

### 11.8 引擎修复（2026-08-02，集成前置）

在接入完整核心前修复两处：

- **未选权重 ROM 越界读取**：两个权重 ROM 原先都以 `issuing` 门控；但 conv3
  运行期间 conv2 ROM 的地址可达到 4751（oc·144 + ic·9 + ky·3 + kx，ic 上到 31），
  超过 conv2 ROM 深度 4608。改为**按层门控**：
  `wt2_read = issuing && !layer`、`wt3_read = issuing && layer`，未选中层的 ROM
  地址固定为 0（非 issuing 周期同样为 0）。`layer` 在 start 时锁存、整次运行稳定。
- **multiplier 十六进制注释勘误**：conv2/conv3 的正确 hex 为 `32'h416335B9` /
  `32'h4D6CC8EC`（与 `baseline_cnn_params.vh` 一致）；原注释 `4162F7B9`/
  `4D6C5DEC` 错误，已修正（十进制 localparam 值本就正确、未改动）。

两个测试向量（`tb_conv2_pool2` / `tb_conv3_serial`）新增层级仿真断言：
conv2 运行期间 conv3 ROM 地址恒为 0、conv3 运行期间 conv2 ROM 地址恒为 0、
任何权重 ROM 不得收到超出自身 DEPTH 的地址。conv2/conv3 全部黄金结果保持不变。

## 12. 流式全局平均池（gap_stream_u8，2026-08-02）

独立可综合的流式 GAP，直接消费 conv3_q 流（32×7×7、channel→y→x、每通道连续
49 项），每通道产出 1 个结果：

```text
sum[c]   = 该通道 49 个 UINT8 之和（14 位无符号寄存器，max 12495）
gap_q[c] = round_half_away_from_zero(sum[c]/49) = floor((sum[c]+24)/49)，饱和到 [0,255]
```

- **复用已冻结的 `gap_div49.v`**（常数除法 `n*2675 >> 17`，穷举验证过，不重实现）；
- 流式契约与 MaxPool 一致：输入计数/累加仅在 `busy && in_valid` 推进，支持任意
  空拍；每 49 项产生一个 `out_valid`；`out_channel` 严格 0..31，共 32 项；
- **与 conv3 同启**（控制器 CONV3_START 同时拉高 `conv_start` 与 `gap_start`），
  不由 conv3_done 启动；start 与第一项 in_valid 天然不同采样沿（conv3 首项 q 在
  start 后 ~290 周期）；
- busy 覆盖 S_DONE（`busy = RUN || DONE`），无 `busy=0 && done=0` 窗口；done
  单周期，且严格在最后一个 out_valid 之后 1 周期；
- 周期（无空拍）：1568 项消费完，done 在最后一项后 2 周期（1 周期到最后一个
  out_valid，1 周期 S_DONE）。实测（tb_gap_stream_u8）：连续 1570、固定空拍 2196，
  busy_cycles 一致。

### 12.1 验证（tb_gap_stream_u8，digit8 / test index 61）

三遍：连续输入（32/32）、**无 reset 连续第二次运行逐位相同**、固定空拍
（每 5 发插 2 空拍，32/32）。gap_q 与 `gap_q.mem` 逐位一致；out_channel 严格
0..31；done 单脉冲；busy 保持到 done；无 X/Z。

## 13. 全连接 + Argmax（fc_argmax_serial，2026-08-02）

序列化 FC 层 + signed argmax。内部保存 `gap_mem[0:31]`（32×UINT8 寄存器阵列，
由 `gap_we/gap_waddr/gap_wdata` 写入）、`fc_weight` ROM（320×SINT8，OI）、
`fc_bias` ROM（10×SINT32）。

数值语义（冻结，`data_format.md` §5.5）：

```text
fc_acc[o]   = Σ_f gap_q[f]·fc_weight[o][f] + fc_bias[o]     # S64 精确累加
fc_acc[o]   = saturate_int32(...)                            # INT32 饱和在 bias 之后
prediction  = signed argmax(fc_acc)，并列取更小类别号
```

- gap_q 按 UINT8、权重按 SINT8 解释，乘积加宽到 17 位 signed；
- bias 在所有 32 次 MAC 结束后一次性加入；**不做 requant**；
- **tie 规则（冻结，PyTorch first-max 语义）**：class0 无条件初始化 `best`；
  后续仅在 `acc32 > best` 时更新，**绝不用 `>=`** —— 并列时保留更早类别；
- 每类别产生一个 `fc_valid/fc_class/fc_acc`，共 10 项、地址 0..9；
- `prediction` 在 done 时有效并**保持到下一次 start**；支持无 reset 连续运行；
- 权重地址 `{oc,f}`（= oc·32+f，位拼接无乘法器）；ROM 采用与卷积引擎相同的
  software-pipeline 读取（`wt_raddr` 指向下一 tap 的 issued 地址）；
- 周期：每类 PROLOGUE(1)+MAC(32)+ADD_BIAS(1)+FC_VALID(1)=35；10 类 = 350，
  +1（S_DONE）= **351** start→done。实测 351。

### 13.1 验证（tb_fc_argmax_serial，digit8 / test index 61）

三遍：golden（gap 写入 32/32 层级读回核对、fc_acc 10/10、prediction=8）、
**无 reset 连续第二次逐位相同**、**人工 tie**（gap[0]=82，使类别 5 与 7 并列
6891 且高于其余类别，argmax 选较小类别 5）。TB 同时用观测到的 10 个 fc_acc
重算 signed argmax 并与 prediction 比对（argmax_bad=0），验证 signed 比较与
tie 规则。

## 14. 完整纯计算核心（baseline_cnn_core，2026-08-02）

端到端整数流水线（无 UART）：

```text
input_q RAM → stem_conv_serial → 流式 MaxPool → pool1_q RAM
→ conv_u8_serial (layer_sel=0) → 流式 MaxPool → pool2_q RAM
→ conv_u8_serial (layer_sel=1) → gap_stream_u8 → gap_mem
→ fc_argmax_serial → prediction
```

### 14.1 控制器与启动规则（冻结）

状态机 `IDLE → STEM_START → STEM_WAIT → CONV2_START → CONV2_WAIT →
CONV3_START → CONV3_WAIT → FC_START → FC_WAIT → DONE`：

- **STEM_START** 只拉高 `stem_start`（stem 引擎与 pool1 MaxPool 在流水线内部同启）；
- **CONV2_START** 同一周期拉高 `conv_start` 与 `pool2_start`（layer_sel=0），
  pool2 不由 conv2_done 启动；等待 pool2_done，**断言 conv2_done 与 pool2_done
  同周期**；
- **CONV3_START** 同一周期拉高 `conv_start` 与 `gap_start`（layer_sel=1），
  GAP 不由 conv3_done 启动；等待 gap_done，**断言 conv3_done 与 gap_done 同周期**；
- **FC_START** 在全部 32 个 gap 值写入 gap_mem 后拉高 `fc_start`；等待 fc_done，
  锁存 prediction。

### 14.2 存储与数据流（冻结）

- **仅允许** input_q RAM（784，stem 引擎内）、pool1_q RAM（3136，流水线内）、
  pool2_q RAM（1568，核心内）、gap_mem（32×8 寄存器，FC 内）；
- **不实例化**完整 stem_q（12544）/ conv2_q（6272）/ conv3_q（1568）RAM：
  stem_q 直入 MaxPool、conv2_q 直入 pool2、conv3_q 直入 GAP；
- conv 引擎的 `fm_raddr` 在 CONV2 阶段读 pool1 RAM、CONV3 阶段读 pool2 RAM；
  每个 RAM 的读地址在非自身阶段被钳到 0（防止 pool2 RAM 被读到 3135 越界），
  数据 mux 按控制器阶段选择，**阶段全程稳定**、中途不切换；
- conv2 的 q 流经参数化 MaxPool 写入 pool2 RAM；conv3 的 q 流**仅在 CONV3 阶段**
  使能进入 GAP（`gap_in_valid = conv_q_valid && conv3_phase`）。

### 14.3 核心级 busy/done 协议（冻结）

`start` 只在 IDLE 接受；busy 覆盖全部非 IDLE 状态（含 DONE），从 start 到 done
脉冲之间无 `busy=0 && done=0` 窗口；done 单周期（DONE 状态的下一个 IDLE 周期）；
done 周期 busy 可为 0；prediction 在 done 时有效并保持到下次 start；一次推理后
无需 reset 即可装载下一张图并再次 start（10 smoke 样本无 reset 连续运行验证）。

### 14.4 周期（digit8 / test index 61 实测）

| 段 | 周期 |
|---|---|
| start → stem_done | 188,162 |
| conv2_start → pool2_done | 940,802 |
| conv3_start → gap_done | 460,994 |
| fc_start → fc_done | 352 |
| start → core_done | **1,590,315** |

各子块比理论值（188,161 / 940,801 / 460,993 / 351）多 1 为控制器阶段计数口径。
端到端值相对旧设计精确增加 `20,384 × 3 = 61,152` 周期：
`1,529,163 + 61,152 = 1,590,315`；50 MHz 下为 **31.8063 ms**。

### 14.5 验证（tb_baseline_cnn_core）

**A. digit8 完整黄金 trace（test index 61）**：`input_q→conv1_acc→stem_q→
pool1_q→conv2_acc→conv2_q→pool2_q→conv3_acc→conv3_q→gap_q→fc_acc→prediction`
逐项 100% 逐位一致（12544/12544、3136/3136、6272/6272、1568/1568、1568/1568、
32/32、10/10、prediction=8）；所有流地址严格连续；控制器 stage 顺序
IDLE..DONE..IDLE 正确；conv2+pool2 同启、conv3+GAP 同启、done 搭档同周期；
GAP 写 FC 恰 32 项、FC 恰 10 项；无 X/Z；无非法 RAM/ROM 地址；done 恰 1 次。

**B. 10 smoke 样本（digit0..digit9，无全局 reset）**：每帧 IDLE 状态重写 784 字节
输入并 start；10 个 fc_acc 与 smoke 黄金逐位一致；prediction == digit（0..9 全对）；
每帧 done 恰 1 次；后帧不残留前帧 pool/gap/argmax 状态（fc_acc 一致即为残留检查）。

### 14.6 综合资源（baseline_cnn_core_smoke，EP4CE10F17C8，2026-08-02 Fitter 实测）

Flow Successful，LE **3,114**、寄存器 **928**、9-bit 嵌入式乘法器 **19**、
**M9K 29 块（63%）**、块内存位 160,512/423,936（38%）、实现位 267,264（63%）、
物理引脚 0、虚拟引脚 69。

M9K 分项（Fitter RAM Summary，最终数字以此为准）：

| 存储 | 模板 | 容量 | M9K |
|---|---|---|---|
| input_q RAM | `sync_ram_u8` | 784×8 = 6272 bit | **1** |
| stem weight ROM | `sync_rom_s8` | 144×8 = 1152 bit | **1** |
| pool1_q RAM | `sync_ram_u8` | 3136×8 = 25088 bit | **4** |
| conv2 weight ROM | `sync_rom_s8` | 4608×8 = 36864 bit | **8** |
| conv3 weight ROM | `sync_rom_s8` | 9216×8 = 73728 bit | **9** |
| conv2 bias ROM | `sync_rom_s32` | 32×32 = 1024 bit | **2** |
| conv3 bias ROM | `sync_rom_s32` | 32×32 = 1024 bit | **2** |
| pool2_q RAM | `sync_ram_u8` | 1568×8 = 12544 bit | **1** |
| fc weight ROM | `sync_rom_s8` | 320×8 = 2560 bit | **1** |
| **fc bias ROM** | `sync_rom_s32` | 10×32 = 320 bit | **逻辑**（10 LC） |

- stem bias 同样落入逻辑（12 LC/12 reg，512 bit 太小）；其余 M9K 分项合计 29；
- **不存在完整 stem_q / conv2_q / conv3_q RAM**（Fitter RAM Summary 只列出上表
  存储，无 12544/6272 深度 RAM，conv3_q 无独立 RAM）；
- 无锁存器、无截断（0 条 Warning 10230）、无实际 signed 警告（77 条匹配均为
  LPM 参数显示/Signed Integer 参数，DSP 乘法器 5 有符号 + 2 无符号）；
- **无除法器/模运算器**；无厂商 IP；
- 本阶段仍未加时钟约束（板载时钟未冻结），Fitter 未做时序收敛
  （"Timing requirements not specified"）。

## 15. A+ Requant 三级流水化（P=3，2026-08-02）

为切断 AC620 50 MHz 下的 `acc64 → requant → pool/GAP` 长组合路径，stem 与共享
conv2/conv3 引擎改用厂商无关的 `rtl/requantize_u8_pipe.v`。原组合
`rtl/requantize_u8.v` 保留不变，继续作为 20,384 项 bit-exact oracle。

### 15.1 固定三级边界与数值语义

```text
SAT32                  MUL                         ROUND_SHIFT_SAT
S64 post-bias ─reg→ S32 × S32 signed ─reg→ S64 ─reg→ magnitude round/shift/U8 sat
```

- `SHIFT` 是编译期参数（stem/conv 均为 38），不是运行时输入；
- multiplier 与 token 在 SAT32 同拍锁存；只推断一个明确的 S32×S32→S64 乘法；
- 负积先取 magnitude，加 half 后逻辑右移，再恢复符号，禁止直接对负数 `>>>`；
- acc32 伴随 MUL/ROUND 两级前推，与 q 在同一 out_valid token 输出；
- `rst_n` 清空全部 valid，数据寄存器无 valid 时保持；可连续每拍接受 token；
- 单元 TB 覆盖 20,384 黄金 token、连续 token、伪随机空拍、中途 reset、正负
  multiplier、INT32 上下界、负 half 边界及 SHIFT=0，结果 `REQUANT_PIPE_ALL_PASS`。

### 15.2 引擎 token 与 FSM

stem/conv 均采用：

```text
IDLE → PROLOGUE → MAC/ACC → ADD_BIAS
     → SAT32 → MUL → ROUND_SHIFT_SAT → EMIT → PROLOGUE/DONE
```

离开 ADD_BIAS 的边沿锁存：

```verilog
bias64_w          = sign_extend_32_to_64(bias_rdata);
post_bias_acc64   = acc64 + bias64_w;
token_acc64      <= post_bias_acc64;
token_addr/last  <= current_addr/last_output;
```

因此不会因同一 always 块中的非阻塞赋值而误用加 bias 前的旧 acc64。通用 requant
模块不携带 metadata；地址/last 在引擎内保持到 EMIT。EMIT、pipe out_valid、
acc_valid、q_valid、acc/q/address 严格同周期；若 pipe valid 未到，FSM 停留 EMIT，
不推进坐标、不写 RAM。无效周期的 acc/q valid、data 和 address 全部钳为 0，局部 TB
同时断言 EMIT 必须与 pipe out_valid 一一对应且无效周期不得泄漏旧 token。stem 的
`STORE_OUTPUT_RAM=1` 也只在 valid-qualified EMIT 写入。

最后一个 EMIT 的 q 被 MaxPool/GAP 在同一边沿消费，producer 与 consumer 同时进入
S_DONE，下一周期仍保持 stem/pool1、conv2/pool2、conv3/GAP co-done。

### 15.3 P=3 周期与功能回归

| 引擎 | first q | start→done | 公式 |
|---|---:|---:|---|
| stem | 14 | 188,161 | `12544 × (1+9+1+4) + 1` |
| conv2 | 149 | 940,801 | `6272 × (1+144+1+4) + 1` |
| conv3 | 293 | 460,993 | `1568 × (1+288+1+4) + 1` |
| 完整核心 | — | **1,590,315** | `1,529,163 + 20,384×3` |

局部 Questa 已通过流水 requant、conv2+pool2、conv3 两遍、真实 conv3+GAP、stem
两遍、padding、stem+pool1 两遍；完整核心 digit8 11 节点与无 reset 连续 digit0..9
全部通过；板级 loader/prediction/LED 自检确认 prediction=8 并输出
`AC620_CNN_SELFTEST_PASS ALL_PASS`。此外 `run_all.ps1` 14/14、完整 pytest 219 passed。

### 15.4 资源、STA 与下载状态

唯一一次 AC620 Quartus 25.1std 完整编译已完成，Flow Successful。实测资源为：

- LE 2,933、寄存器 1,243、M9K 30、memory bits 166,784；
- 9-bit 乘法器元素 19、DSP blocks 11、PLL 0；
- 物理引脚 5、虚拟引脚 0。

20.000 ns 约束下所有分析 corner 均通过，setup/hold 分别为：Slow 85C
`+2.274/+0.429 ns`、Slow 0C `+3.853/+0.400 ns`、Fast 0C
`+12.380/+0.150 ns`；最小 Fmax 为 **56.41 MHz**。retiming 前 Slow 85C 的
`conv_u8_serial.acc64[42] → gap_stream_u8.out_q_r[6]`、`-16.968 ns` 长路径已
消失；新最差 setup 路径为 conv3 权重 ROM 地址寄存器到
`conv_u8_serial.acc64[63]`，slack `+2.274 ns`。

因此该实现已达到 **50 MHz timing-qualified**。编译生成的 `.sof` 来自尚未提交的
dirty worktree，只作为本地工程验证证据，不是正式可追溯烧录件；提交前审查完成并
再次获得用户明确确认前不得 JTAG 下载。本记录不宣称已在真实 AC620 上运行。

收尾后的 `run_ac620_cnn_selftest.ps1 -ValidateExistingReportsOnly` 可在不调用 Questa、
Quartus compile 或 Programmer 的情况下复核现有报告：要求 Flow Successful、关键层级
（含 `requantize_u8_pipe`）、资源下界、六条 setup/hold 且 slack 非负/TNS 为 0，以及
非空 SOF。缺层级、M9K/DSP 降低或负 setup 的副本负向用例均能以非零退出。
