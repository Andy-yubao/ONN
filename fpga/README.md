# `fpga/`：正式模型的硬件实现

`fpga/` 实现 `model/` 中冻结的部署候选；当前 seed 17 SNN 尚未提升为 production champion。
已完成 SNN RTL Phase 1–4：
基础数值单元、Conv1+IF1、Conv2+IF2、readout、四步时间加权、argmax 和完整 `snn_core`。
部署顶层现为 spike-driven sparse：只对实际 spike 执行突触权重累加；原 dense RTL 保留为
黄金基线。两种路径均由 Python integer reference 做逐层和端到端逐 bit 验证。

AC620 / Cyclone IV 属于 legacy platform；当前新的优先候选目标板是 Basys3 / Artix-7。
[`basys3/`](basys3/) 已增加 USB-UART packet 协议、PC host/mock、wrapper、XDC 与
脚本化 Vivado 流程，并完成串口波形到 core 再到结果包的仿真。综合与实现结论以
[`basys3/verification_record.md`](basys3/verification_record.md) 及正式报告为准。
2026-09-23 已完成首次真实 Basys3 下载和 USB-UART 对拍：4 个人工输入及
100 张 validation 样本与整数参考逐项一致；真实器件输入与实测功耗仍未运行。

目录职责：

- `rtl/`：事件驱动部署顶层、dense 黄金基线与数值/接口契约；
- `tb/`：各层独立 testbench 和完整四步 `snn_core` testbench；
- `vectors/`：由共享 Python integer reference 生成的确定性黄金向量；
- `scripts/`：详细/紧凑向量生成与 Vivado Simulator 自动化入口；
- `basys3/`：当前板级通信和综合/实现准备。

在仓库根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File fpga/scripts/run_rtl_regression.ps1
```

脚本使用固定的 `D:\tools\anaconda3\envs\onn\python.exe`，重新生成向量后运行
`xvlog/xelab/xsim`，依次运行 IF、dense 各层、完整 sparse core、48 帧紧凑差分回归和
UART wrapper 波形仿真。端到端测试核对 hidden spike、十个分数、类别、每步 sparse
突触加法数和周期数；任一数值、计数或 packet 字节比较失败都会返回失败。
