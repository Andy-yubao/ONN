# 当前项目范围

> 本文档是当前范围的唯一事实源。旧 28×28 CNN / AC620 成果由 Git tag 保存，摘要见
> [`history/README.md`](../history/README.md)，不再作为 active tree 的开发主线。

## 一句话定义

ONN 当前研究在统一的 8×8 输入和器件动态编码协议下，建立参数量相近的 CNN 与 SNN
候选，进行公平比较；若 SNN 在准确率、事件计算、器件编码耦合和 FPGA 资源之间形成
可接受折中，再将正式 champion promote 到 `model/` 并部署到 `fpga/`。

## 当前主线

```text
8×8 输入 / 器件动态模型
        ↓
device latency / first-spike encoding
        ↓
参数量相近的 CNN vs SNN 公平实验
        ↓
完成 champion selection（若选择 SNN，再 promote）
        ↓
model/：量化、整数参考、硬件导出
        ↓
fpga/：RTL、仿真、综合与板级验证
```

统一 8×8 matched CNN 已完成，held-out test mean 97.37%，高于 Frozen SNN 的
93.47%。因此不声称 SNN 优于 CNN，也不声称其满足尚未冻结的产品精度要求。
当前 seed 17 SNN deployment checkpoint 是硬件研究用的冻结对象，尚非 production champion。

## 已记录结果与边界

三项保留的 SNN 实验及结果见 [`experiments/README.md`](../experiments/README.md)：

- MLP latency baseline：87.15%，训练时逐轮查看 test set，属于 exploratory 结果；
- large Conv-SNN architecture exploration：91.27%，因逐轮查看 test set，不作为严格
  独立 held-out test benchmark；
- small Conv-IF-SNN（原 `snn_8x8_device_if_conv_small`）后续完成 Frozen T=4 三 seed
  公平实验，held-out test 为 93.47% ± 0.14 pct，9,872 参数；
- matched 8×8 CNN 为 97.37% ± 0.11 pct，9,930 参数。完整公平比较见
  [`final_three_seed_record.md`](../experiments/snn/conv_small/records/final_three_seed_record.md)。

Frozen SNN 的 INT8 PTQ、guard-4 整数参考、seed 17 参数导出均已完成；
三 seed 整数 reference test mean 为 93.24% ± 0.17 pct。
事件驱动 sparse RTL 已与整数参考完成逐层和端到端仿真对拍，详见
[`quantization_record.md`](../model/snn/quantization_record.md) 和
[`fpga/README.md`](../fpga/README.md)。Basys3 的 UART、host、仿真、综合、
布局布线和 bitstream 已完成；结果见
[`synthesis_summary.md`](../fpga/basys3/synthesis_summary.md)。首次真实 FPGA 对拍结果见
[`verification_record.md`](../fpga/basys3/verification_record.md)。

## 四个核心目录

| 目录 | 唯一职责 |
|---|---|
| `experiments/` | 候选方案、失败实验、消融实验、公平比较 |
| `model/` | 正式选定模型、量化、整数参考、部署导出 |
| `fpga/` | `model/` 冻结部署候选及未来正式模型的 FPGA 实现 |
| `history/` | 历史阶段索引与 Git tag 入口，不复制旧源码 |

## 硬件状态

- AC620 / Cyclone IV 是 legacy platform；
- 当前优先候选目标板是 Basys3 / Artix-7；
- seed 17 部署数值格式和位宽已冻结；Basys3 综合、布局布线、静态时序和 bitstream 已完成；
- Basys3 wrapper、约束和脚本化 Vivado 工程用于当前部署验证，不代表已确认产品目标；
- 不把旧 AC620 固定 digit8 自检扩展为当前 SNN 或任意输入部署证据。

## 本轮边界

不重新训练、不重新选择 champion、不修改 INT8/guard-4/threshold/时间系数契约。
第一次真实下板只验证 `4×64-bit spike → SNN core → result`，不接入真实 ADC 或器件。
软件仿真、综合与实现结果需与真实 FPGA 通信和推理结果分别报告。
