# `model/`：量化与部署准备

当前已固定 Frozen T=4 SNN 的唯一 **RTL deployment checkpoint**，用于后续 SNN 硬件实现；
这不等同于将 SNN 选为项目 production champion，也不改变 matched CNN 准确率更高的结论。

本目录承载部署契约：

- frozen architecture；
- trained checkpoint metadata；
- quantization；
- fixed-point / integer reference；
- hardware parameter export；
- deployment-oriented tests。

## Frozen SNN PTQ

- [量化记录与硬件位宽表](snn/quantization_record.md)
- [结构化实测结果](snn/results/ptq_summary.json)
- [内部状态压缩实测结果](snn/results/state_quantization_summary.json)
- [整数 / 定点 reference](snn/integer_reference.py)
- [PTQ、calibration 与完整测试入口](snn/ptq.py)
- [内部状态位宽选择入口](snn/state_quantization.py)
- [数值契约测试](snn/test_integer_reference.py)
- [唯一 RTL checkpoint 与选择依据](snn/deployment_checkpoint.md)
- [硬件参数导出器](snn/export_params.py)
- [硬件参数与机器可读 manifest](snn/export/manifest.json)
- [导出读回与契约测试](snn/test_export_params.py)

当前已完成三 seed PTQ 与整数 reference test inference，并依据 validation 和确定性复现固定
seed 17 为唯一 RTL checkpoint；INT8 weight、guard-4/INT17 位宽、整数 threshold 和时间系数
已导出。下游已完成事件驱动稀疏 SNN RTL 和 Python reference 的仿真级逐 bit 对拍；尚未创建
最终 Vivado 工程、运行 synthesis / implementation 或进行 FPGA 板级验证。RTL 状态与入口
见 [`../fpga/README.md`](../fpga/README.md)。
