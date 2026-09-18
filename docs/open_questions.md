# 开放问题

本文件只记录当前 8×8 CNN/SNN 与后续 FPGA 主线的问题；已冻结的旧时代问题不在这里
继续维护。

## 待实验确定

- matched 8×8 CNN 的最终架构与准确率；
- CNN 与当前最重要的 `experiments/snn/conv_small/` 基线（原
  `snn_8x8_device_if_conv_small`）在统一协议下的准确率/资源折中；
- effective event/synaptic operations 与状态访问量的可比记录方式；
- SNN champion 的稳定性、位宽需求和溢出边界。

## 待设计确定

- champion 的权重、膜电位和时间读出位宽；
- BRAM 权重组织与 DSP/adder reuse 方案；
- temporal controller 的时钟/逻辑时间映射；
- Basys3 的输入接口（UART/PC 或其他方式）与板级验证协议；
- 是否需要在正式硬件前冻结某个可复现的输入向量格式。

## 当前已确认

- 输入主线为 8×8 MNIST，经器件动态模型进行 first-spike latency encoding；
- 共享编码参数为 `G0=0.10`、`alpha=0.90`、`tau=5.0`、`G_threshold=0.35`、`T=24`；
- 当前共同实现位于 `experiments/common/device_latency_encoder.py`；
- AC620 / Cyclone IV 只作为 legacy platform；Basys3 / Artix-7 是当前候选目标；
- matched CNN 完成并选出 champion 前，不开始正式 SNN 量化或 FPGA RTL。
