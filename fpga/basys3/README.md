# Basys3 / Artix-7 下板准备

本目录实现 seed 17 冻结 SNN core 外围的 USB-UART 输入、结果输出和
Basys3 约束。板级接口只接收 `4×64-bit` spike；真实 ADC 和器件输入不在此次首次下板范围。
数值 core 保持在 [`../rtl/`](../rtl/)；通信格式见 [`uart_protocol.md`](uart_protocol.md)。

- [`rtl/`](rtl/)：UART RX/TX、packet parser、四步控制器与结果发送；
- [`host.py`](host.py)：MNIST 8×8 预处理、冻结 encoder、packet、mock/串口验证 CLI；
- [`basys3.xdc`](basys3.xdc)：100 MHz 输入、USB-UART、按键及四个 LED 约束；
- [`docs/`](docs/)：从本机已有资料复制的 Digilent Master XDC 和 Basys3 手册；
- [`vivado/run.tcl`](vivado/run.tcl)：脚本化建立工程并运行 synthesis / implementation。

所有 Python 入口均从仓库根目录、用 `D:\tools\anaconda3\envs\onn\python.exe` 运行。
没有串口设备时先运行：

```powershell
& 'D:\tools\anaconda3\envs\onn\python.exe' -m fpga.basys3.host --backend mock --level artificial
& 'D:\tools\anaconda3\envs\onn\python.exe' -m fpga.basys3.host --backend mock --level golden
powershell -ExecutionPolicy Bypass -File fpga/scripts/run_rtl_regression.ps1
```

回归通过后，按顺序运行 Vivado：

```powershell
powershell -ExecutionPolicy Bypass -File fpga/basys3/vivado/run.ps1 -Stage synth
powershell -ExecutionPolicy Bypass -File fpga/basys3/vivado/run.ps1 -Stage impl
```

脚本使用单进程批处理流程，避免本机 Vivado 工程子进程启动时报出的
`rundef.js Access denied`；同目录的 `run.tcl` 仍可重复创建 `.xpr` 项目。
MMCM 将板载 100 MHz 输入转换为 25 MHz 内部时钟，逻辑和 UART 共用此时钟，
不引入两个逻辑时钟域。综合与实现报告位于 `reports/`，不得以脚本启动代替检查报告和 DRC。
本次已生成 [`reports/onn_basys3.bit`](reports/onn_basys3.bit)，布线后的时序和资源记录见
[`synthesis_summary.md`](synthesis_summary.md)。

连接真实板卡后运行以下命令，经 Vivado Hardware Manager 向唯一的 `xc7a35t`
器件下载上述 bitstream 并检查 DONE、CRC 和 IDCODE 状态。板卡断电后须重新下载；
完成后按 `btn_c` 复位，确认串口号，再按下列顺序运行：

```powershell
powershell -ExecutionPolicy Bypass -File fpga/basys3/vivado/run.ps1 -Stage program
```
串口 backend 需要在 `onn` 环境中安装
[`requirements-serial.txt`](requirements-serial.txt) 中的 `pyserial`；当前 mock/RTL
仿真不依赖它。

```powershell
& 'D:\tools\anaconda3\envs\onn\python.exe' -m fpga.basys3.host --backend serial --port COM3 --level artificial
& 'D:\tools\anaconda3\envs\onn\python.exe' -m fpga.basys3.host --backend serial --port COM3 --level golden
& 'D:\tools\anaconda3\envs\onn\python.exe' -m fpga.basys3.host --backend serial --port COM3 --level batch --indices 55000 55001 55002
```

`COM3` 仅为命令示例，实际端口需按本机设备确定。`--split test` 只用于最终评价；
日常对拍优先使用 validation。输出将 prediction/score/counter mismatch 与 accuracy 分开统计。
也可用 [`validate_board.ps1`](validate_board.ps1) 按 Level 1→2→3 顺序执行，
例如 `-Port COM3 -Level 3 -Count 100`。
2026-09-23 的首次实板验证已完成：人工输入 4/4、validation golden 2/2、
validation 批量 100/100 均与整数参考逐项一致。断电后重新下载 bitstream 也通过，
具体结果见 [`verification_record.md`](verification_record.md) 和
[`results/board_validation_20260923/`](results/board_validation_20260923/)。
真实器件输入、ADC 接入和实测功耗仍未运行。
