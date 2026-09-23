# Basys3 验证记录

下板前验证日期：2026-09-21；首次实板验证日期：2026-09-23。目标：seed 17
Frozen T=4 SNN，Basys3 XC7A35T-1CPG236C，USB-UART 输入四个 64-bit spike bitmap。

## 资料与冻结边界

- 数值黄金参考：`model/snn/integer_reference.py`，未修改 INT8、guard-4、threshold
  或 `[5,4,3,2]` 时间系数；
- 本机 Basys3 Reference Manual 和 Master XDC 已检查，复制件 SHA256 与原件一致；
- 使用官方 XDC 对应的 W5 时钟、B18/A18 USB-UART、U18 按键与四个 LED 引脚；
- PC 与 FPGA 的格式、字节序、bit 顺序及错误行为见 [`uart_protocol.md`](uart_protocol.md)。

## 已运行验证

| 项目 | 结果 |
|---|---|
| Python 基础测试 | 17 passed：整数参考、参数导出、packet codec、共享 encoder |
| 原有 RTL 逐层 | IF 144、Conv1 10×1024、Conv2 7×512、readout 7×10，均通过 |
| 原有详细 core trace | 4 帧/16 timestep，L1/L2/readout 流与整数参考一致 |
| 新增紧凑 core regression | 48 帧/192 timestep，L1/L2 spike bitmap、十个最终与逐步 score、class、sparse additions、cycle count 全部一致 |
| 紧凑样本构成 | 8 个人工 corner，MNIST train split 的 validation indices 55000..55031 共 32 张，固定 seed 170421 的 8 组随机 bitmap |
| UART wrapper 波形 | 7 个完整请求加 1 个不完整请求；7 个完整响应逐字节相同，覆盖零输入、两张 MNIST、连续帧、busy 丢弃、checksum/index 错误、reset 和不完整包超时 |
| DEBUG_BUILD 波形 | 一张 MNIST validation 55002 的 73-byte 扩展结果逐字节匹配，包含四步 L1/L2 spike count |
| PC host mock | 人工 4 帧与 validation 55000..55009 共 10 张，score/class/counter 全部 PASS |

紧凑 vectors 只保存输入 bitmap、L1/L2 spike bitmap、十个分数及计数，详细 L1/L2
膜电位 trace 仍只保存少量样本。样本来源和 index 见
[`../vectors/regression_manifest.json`](../vectors/regression_manifest.json)。
随机测试使用 Python `random.Random(170421)`，产生每 timestep 独立 64-bit bitmap，
允许同一位置跨 timestep 重复发放。该组测试验证 RTL 与整数参考的一致性，
不作为 MNIST accuracy 结论。

mock transport 使用同一整数参考构造响应，仅验证 host/codec/validator 逻辑；
它不是 FPGA 推理结果。真实板级结果单独记录于下文。

## Vivado 与板级状态

Vivado 2026.1 BASIC 许可证与目标器件可用。脚本见 [`vivado/`](vivado/)；
器件为 `xc7a35tcpg236-1`。100 MHz 直接驱动的首次综合未过时序，
最差路径在 sparse IF 状态更新。当前采用板载 100 MHz 输入经 MMCM 生成
25 MHz 内部时钟，UART 和 core 同域；Conv1 IF 数据通路做了等价化简，
完整 RTL 回归重新通过。当前综合 WNS +22.552 ns、TNS 0、DRC 零项。
implementation 已完成，顶层 10,762 LUT（其中 1,516 LUTRAM）、6,283 FF、
0 BRAM、0 DSP；布线后 WNS +16.128 ns、TNS 0、最差 hold slack +0.014 ns，
setup/hold 无失败 endpoint，DRC 零项。`onn_basys3.bit` 已生成。
资源映射、并行度判断及报告入口见 [`synthesis_summary.md`](synthesis_summary.md)。
首次下板步骤见 [`README.md`](README.md)。

## 真实 Basys3 验证（2026-09-23）

Vivado Hardware Manager 通过 Digilent JTAG 识别到 `xc7a35t_0`，下载上述
`onn_basys3.bit`。用户中途断电后重新下载，刷新设备后配置状态为 `DONE=1`、
CRC error=0、IDCODE error=0。Windows 将同一板载 FTDI USB-UART 识别为 COM8；
该端口号只适用于本次主机连接。串口使用 115200 baud。

主机验证使用 `onn` 环境的 Python；`pyserial==3.5` 安装在工作区忽略的
`.vivado/pydeps`，运行时通过 `PYTHONPATH` 加载。三轮均使用 serial backend，
并逐帧核对 prediction、十个 score、synaptic-add count 与 compute-cycle count：

| 级别 | 输入 | 对拍 | 通信错误 | 分类正确数 |
|---|---|---:|---:|---:|
| 1 | 4 个人工输入：零、单 spike、角点、密集 | 4/4 | 0 | 无标签 |
| 2 | MNIST validation 55002、55003 | 2/2 | 0 | FPGA/参考均为 2/2 |
| 3 | MNIST validation 55000..55099 | 100/100 | 0 | FPGA/参考均为 95/100 |

三轮 prediction、score、两个 counter 的 mismatch 均为 0。原始主机 JSON 见
[`results/board_validation_20260923/`](results/board_validation_20260923/)；第 2 级样本
包含在第 3 级的 100 张中。95/100 是这 100 张 validation 样本的分类结果，
不是 held-out test 准确率，也不代表已选定 production champion。真实器件输入、
ADC 接入和 FPGA 实测功耗仍未运行。
