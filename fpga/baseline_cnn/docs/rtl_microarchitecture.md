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
