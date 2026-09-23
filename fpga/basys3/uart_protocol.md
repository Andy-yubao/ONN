# Basys3 SNN UART 协议 v1

此协议只承载一帧四个 64-bit spike bitmap 及其推理结果。物理链路为板载 USB-UART，
115200 baud、8 数据位、无校验位、1 停止位、无硬件流控。板上输入时钟为 100 MHz，
MMCM 为逻辑和 UART 生成 25 MHz 时钟；UART 每 bit 使用 217 个内部时钟。
PC 一次只发送一个请求，并等待对应序号的结果后再发送下一帧。

## 通用封装

| 偏移 | 字节 | 含义 |
|---:|---:|---|
| 0–1 | `A5 5A` | 同步字 |
| 2 | `01` | 协议版本 |
| 3 | `00..FF` | sequence，原样回显，回卷允许 |
| 4 | 见下文 | packet type |
| 5 | 见下文 | payload 字节数 |
| 6.. | payload | 固定长度，由 type 决定 |
| 末尾 | XOR | 从版本字节到 payload 最后一字节逐字节异或 |

多字节数值均为 **little-endian**。有符号分数以 32-bit 二补码传输，
由 RTL 的 signed INT21 分数符号扩展而来。输入 bitmap 的 bit `y*8+x` 对应像素 `[y,x]`；
在串口线上，低位字节先发送，字节内 UART 也是最低位先发送。

## 请求 `type=01`、`length=36`

payload 顺序固定为四组 `(timestep index: 1 byte, spike bitmap: 8 bytes)`。
index 必须依次为 0、1、2、3。index 0 隐含 `frame_start=1`，后续为 0。
同一像素可以在多个 timestep 发放，以便测试 RTL 状态行为；正常 MNIST encoder
只在一个 timestep 发放，未发放像素对应四个 bitmap 中均为 0。

## 结果 `type=81`、`length=50`

| payload 偏移 | 字节数 | 含义 |
|---:|---:|---|
| 0 | 1 | status：0 成功，1 checksum 错，2 header/长度/index 错，3 busy |
| 1 | 1 | predicted class，0..9 |
| 2 | 40 | 类别 0..9 的十个 final weighted score，各 signed 32-bit |
| 42 | 4 | 整帧 sparse synaptic-add count，unsigned 32-bit |
| 46 | 4 | 整帧 core compute-cycle count，unsigned 32-bit |

cycle count 是四个 timestep 的 L1、L2 和 readout 计算周期之和；不含 UART
传输、wrapper 调度和顶层交接周期。非零 status 时，class、scores、两个计数均置 0。
PC 应同时检查 sequence、status、十个 score、class 和两个计数，不能只看 accuracy。

`DEBUG_BUILD=1` 时，成功结果改用 `type=82`、`length=66`，在上述 50 字节后
追加四组 L1/L2 spike count，各 16-bit little-endian。错误响应仍使用 `81`。
正常构建不经 UART 传输逐元素 trace。

## 错误与重同步

接收端查找同步字，验证版本、固定 type/length、timestep index 和 XOR。
完整请求的 XOR 失败返回 status 1；type、length 或 index 不合法返回 status 2。
收到不完整包且字节间空闲达到 100 ms 后返回 status 2（若 sequence 已接收），并重新查找同步字。
错误请求不会启动 SNN core。孤立字节或错误同步字被丢弃；完整错误包或超时后，
收到下一个 `A5 5A` 可重新同步。板上 `btn_c` 为高时复位 UART、parser、controller 和 SNN 状态；
释放后可直接发送新帧。

wrapper 只有一帧执行槽和一个响应发送槽。处理推理或发送结果时收到的完整新请求
被丢弃，不影响已接收帧；不会发额外响应。PC 必须串行发送并等待结果，超时后可
按 `btn_c` 复位再重试。`status=3` 为协议预留值，当前单槽 wrapper 不发送该值。
本协议的 XOR 是传输错误检测，不提供纠错或安全认证。

PC codec 与 CLI 见 [`protocol.py`](protocol.py)、[`host.py`](host.py)。
