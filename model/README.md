# `model/`：量化与部署准备

当前仍未选择 production / deployment champion。这里先承载 Frozen T=4 SNN 的 PTQ 和
硬件导向整数参考，身份是 **deployment candidate**，不改变 matched CNN 准确率更高、
production champion 尚未冻结的项目结论。

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

当前已完成三 seed PTQ 与整数 reference test inference；尚未选择供 RTL 导出的单一
checkpoint，也未写 RTL、创建 Vivado 工程或进行 FPGA 验证。
