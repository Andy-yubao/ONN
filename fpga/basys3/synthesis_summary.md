# Basys3 首次综合与架构判断

日期：2026-09-21。器件：`xc7a35tcpg236-1`，Vivado 2026.1，正常模式
`DEBUG_BUILD=0`。本报告记录下板前的综合证据；真实 FPGA 运行仍需板级验证。

## 时钟和资源

板载 `W5` 输入是 100 MHz。MMCM 倍频到 800 MHz 后分频，给 core、UART 和
wrapper 提供同一个 25 MHz 内部时钟。没有 core 与 UART 之间的跨时钟域传输。
100 MHz 直接驱动时，首次综合最差 setup slack 为 -7.560 ns；化简 Conv1 IF
数据通路后仍为 -7.366 ns（最差路径转到 Conv2）。保留已验证的单 lane
数据通路并使用 25 MHz 内部时钟后，综合 WNS 为 +22.552 ns、TNS 为 0，
零 setup/hold 违例。实际可运行时钟仍以布局布线后的报告为准。

| 范围 | 总 LUT | 其中 LUTRAM | FF | RAMB18/36 | DSP |
|---|---:|---:|---:|---:|---:|
| 顶层合计 | 10,972 | 1,516 | 6,282 | 0 | 0 |
| SNN core | 10,233 | 1,516 | 5,288 | 0 | 0 |
| Conv1 + IF1 | 1,658 | 896 | 246 | 0 | 0 |
| Conv2 + IF2 | 4,354 | 592 | 2,180 | 0 | 0 |
| Readout | 2,391 | 28 | 1,186 | 0 | 0 |

权重文件已由 `$readmemh` 成功载入。Vivado 把三个只读权重数组映射为 LUT ROM，
把 current/membrane 状态映射为分布式 RAM；没有使用 BRAM 或 DSP。大面积主要
来自稀疏 core 的权重选择、scatter 寻址和状态逻辑。含 `/`、`%` 的常数寻址
写法已综合进入上述资源，不存在未综合的行为级逻辑。该结构的 LUT 占用较高，
若以后追求更大模型或并行度，应优先研究 BRAM ROM 和同步状态读取的结构。

## 并行度选择

第一次下板保留 **1 lane**。48 帧确定性回归的 core 计算周期范围为
7,934–239,299 周期，在 25 MHz 下约 0.32–9.57 ms；上界来自密集人工/随机
spike，而正常 MNIST 通常更稀疏。UART 收发一帧另需数毫秒，此阶段的首要目标
是逐 bit 对拍。2 lane 需要增加读取端口、scatter/IF 计算和相关寻址结构，
4 lane 的代价更高；现有综合证据没有要求为首次下板承担这些改动。
这只是工程判断，没有把 2/4 lane 写成已综合的实现或实测资源数。

原始报告：[`reports/synthesis_utilization.rpt`](reports/synthesis_utilization.rpt)、
[`reports/synthesis_timing.rpt`](reports/synthesis_timing.rpt)、
[`reports/synthesis_drc.rpt`](reports/synthesis_drc.rpt)。综合 DRC 为零项。

## Implementation 与 bitstream

`opt_design`、`place_design`、`phys_opt_design`、`route_design` 均已完成。
布线后的顶层占用为 10,762 LUT（其中 1,516 LUTRAM）、6,283 FF、0 BRAM、
0 DSP。25 MHz 内部时钟的 WNS 为 +16.128 ns，TNS 为 0；最差 hold slack
为 +0.014 ns，setup/hold 均无失败 endpoint。implementation DRC 为零项，
bitstream 的前置 DRC 也是零错误。Vivado 已成功生成
[`reports/onn_basys3.bit`](reports/onn_basys3.bit)（2,192,217 bytes，SHA256
`532564C7BAD75C545D0A29605F0C573B4E95C62EE5A30208C611654A2001A18F`）。脚本仅在布线后 setup
slack 非负时调用 `write_bitstream`。

原始报告：[`reports/implementation_utilization.rpt`](reports/implementation_utilization.rpt)、
[`reports/implementation_timing.rpt`](reports/implementation_timing.rpt)、
[`reports/implementation_drc.rpt`](reports/implementation_drc.rpt)。
这说明约束下的实现满足静态时序；还没有实板运行或 USB-UART 通信结果。
