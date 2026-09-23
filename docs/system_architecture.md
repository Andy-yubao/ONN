# 当前架构

## 研究到部署的 promotion pipeline

```text
experiments/
    ├─ 器件 latency 编码与 SNN 候选
    ├─ matched 8×8 CNN baseline
    └─ 公平比较与消融
              ↓
      matched CNN/SNN 公平比较已完成；production champion 尚未冻结
              ↓
model/
    seed 17 冻结部署 checkpoint → INT8 / guard-4 → integer reference → export
              ↓
fpga/
    sparse RTL → 逐 bit simulation → Basys3 UART wrapper → synthesis / implementation → board validation
```

`experiments/` 的结果不是 production 承诺。当前 matched CNN held-out test mean
97.37%，Frozen SNN 为 93.47%；SNN 的 seed 17 checkpoint 用于可重复部署研究，
尚未成为 production champion。部署格式和 RTL 逐 bit 仿真验证已完成。

## 当前 SNN 数据流

```text
MNIST 28×28
  → 双线性 Resize 至 8×8
  → G(t)=G0 + alpha·p·(1-exp(-t/tau)) 器件动态模型
  → 24 步 first-spike latency encoding，再经固定 Quantile 边界映射到 T=4
  → frozen seed 17 SNN 部署 checkpoint
  → 分类与事件/状态工作量记录
```

当前共同器件参数为 `G0=0.10`、`alpha=0.90`、`tau=5.0`、`G_threshold=0.35`，
原始编码窗口 `T=24`；部署使用 Frozen T=4 Quantile mapping。共享实现位于
[`experiments/common/device_latency_encoder.py`](../experiments/common/device_latency_encoder.py)。
无阈值事件返回 `-1`，强度越大时 first spike 不会更晚。

## CNN/SNN 公平性

matched CNN/SNN 已使用统一 8×8 输入、55k/5k/10k 数据划分、相近参数规模和
test-set-only-final-evaluation 规则。对比见
[`final_three_seed_record.md`](../experiments/snn/conv_small/records/final_three_seed_record.md)。
准确率、事件计算、资源与功耗是不同指标；不把 GPU 时间当作 FPGA 性能结论。

## 硬件边界

Basys3 / Artix-7 是当前部署验证目标。INT8 权重、guard-4 状态、INT17 readout current、
INT21 weighted score 与时间系数 `[5,4,3,2]` 已冻结。`fpga/rtl/` 保存 dense 黄金基线
和事件驱动 sparse core；`fpga/basys3/` 保存 UART packet 协议、host 工具、wrapper、XDC
和脚本化 Vivado 工程。第一次板级验证只输入四个 64-bit spike bitmap，并对拍十个最终分数、
类别和计数。首次真实 FPGA 板级对拍已完成，结果见
[`../fpga/basys3/verification_record.md`](../fpga/basys3/verification_record.md)。
