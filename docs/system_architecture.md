# 当前架构

## 研究到部署的 promotion pipeline

```text
experiments/
    ├─ 器件 latency 编码与 SNN 候选
    ├─ matched 8×8 CNN baseline
    └─ 公平比较与消融
              ↓
      选择 champion（尚未发生）
              ↓
model/
    frozen architecture → quantization → integer reference → export
              ↓
fpga/
    Basys3 / Artix-7 RTL → simulation → synthesis → board validation
```

`experiments/` 的结果不是 production 承诺。只有统一数据划分、参数规模和评价边界的
CNN/SNN 比较完成后，才会将一个模型放入 `model/`。

## 当前 SNN 数据流

```text
MNIST 28×28
  → 双线性 Resize 至 8×8
  → G(t)=G0 + alpha·p·(1-exp(-t/tau)) 器件动态模型
  → 24 步 first-spike latency encoding
  → candidate SNN
  → 分类与事件/状态工作量记录
```

当前共同编码参数为 `G0=0.10`、`alpha=0.90`、`tau=5.0`、`G_threshold=0.35`、
`T=24`；共享实现位于 [`experiments/common/device_latency_encoder.py`](../experiments/common/device_latency_encoder.py)。
无阈值事件返回 `-1`，强度越大时 first spike 不会更晚。

## CNN/SNN 公平性

下一轮 matched CNN 与 small Conv-IF-SNN 至少共享 8×8 输入、预处理原则、55k/5k/10k
数据划分、seed policy、相近参数规模和 test-set-only-final-evaluation 规则。比较同时
报告准确率、参数/存储、稠密等效操作、有效事件突触操作、状态存储和预期 FPGA 资源，
不把 GPU 时间直接当作 FPGA 性能结论。

## 硬件边界

Basys3 / Artix-7 只是当前候选平台，不是已冻结目标。后续需要在 champion 选定后冻结
权重/膜电位/读出位宽、BRAM 组织、DSP/adder reuse、temporal controller、输入接口和
板级验证协议。当前没有 SNN RTL、Vivado project、约束文件或 bitstream。
