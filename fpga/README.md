# `fpga/`：正式模型的硬件实现

`fpga/` 只用于实现 `model/` 中已正式选定的 champion。当前已完成 SNN RTL Phase 1–4：
基础数值单元、Conv1+IF1、Conv2+IF2、readout、四步时间加权、argmax 和完整 `snn_core`。
部署顶层现为 spike-driven sparse：只对实际 spike 执行突触权重累加；原 dense RTL 保留为
黄金基线。两种路径均由 Python integer reference 做逐层和端到端逐 bit 验证。

AC620 / Cyclone IV 属于 legacy platform；当前新的优先候选目标板是 Basys3 / Artix-7。
当前仍未执行 Vivado synthesis / implementation、资源/时序分析或 FPGA 下板，因此不宣称
硬件部署已经完成。

目录职责：

- `rtl/`：事件驱动部署顶层、dense 黄金基线与数值/接口契约；
- `tb/`：各层独立 testbench 和完整四步 `snn_core` testbench；
- `vectors/`：由共享 Python integer reference 生成的确定性黄金向量；
- `scripts/`：向量生成与 Vivado Simulator 自动化入口。

在仓库根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File fpga/scripts/run_rtl_regression.ps1
```

脚本使用固定的 `D:\tools\anaconda3\envs\onn\python.exe`，重新生成向量后运行
`xvlog/xelab/xsim`，依次运行 IF、dense 各层以及完整 sparse core。端到端测试还会核对
每步 sparse 突触加法数和周期数；任一数值或工作量比较失败都会返回失败。
