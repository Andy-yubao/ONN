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

当前主线不预先断言 SNN 优于 CNN，也不声称准确率已满足目标。统一 8×8 CNN baseline
完成前，SNN 结果只属于候选实验。

## 已记录结果与边界

三项保留的 SNN 实验及结果见 [`experiments/README.md`](../experiments/README.md)：

- MLP latency baseline：87.15%，训练时逐轮查看 test set，属于 exploratory 结果；
- large Conv-SNN architecture exploration：91.27%，因逐轮查看 test set，不作为严格
  独立 held-out test benchmark；
- small Conv-IF-SNN（原 `snn_8x8_device_if_conv_small`）是当前最重要的 SNN 基线：
  best validation 92.32%，final held-out test 90.22%，9,872 参数。

这些数字本次未重跑、未重算。95% 目标尚未达到；matched 8×8 CNN 尚未完成；production
SNN 尚未选定；当前 SNN 尚未量化、导出或实现 RTL。后续公平比较和 champion 选择优先
围绕上述 `conv_small` 基线展开，但不得因此预先宣布 SNN 获胜。

## 四个核心目录

| 目录 | 唯一职责 |
|---|---|
| `experiments/` | 候选方案、失败实验、消融实验、公平比较 |
| `model/` | 正式选定模型、量化、整数参考、部署导出 |
| `fpga/` | `model/` 正式模型的 FPGA 实现 |
| `history/` | 历史阶段索引与 Git tag 入口，不复制旧源码 |

## 硬件状态

- AC620 / Cyclone IV 是 legacy platform；
- 当前优先候选目标板是 Basys3 / Artix-7；
- hardware target 尚未最终 freeze；
- champion、数据位宽、状态存储和资源预算冻结前，不创建最终 Vivado 工程；
- 不把旧 AC620 固定 digit8 自检扩展为当前 SNN 或任意输入部署证据。

## 当前不做

当前尚未安排重新训练、完整 MNIST 重跑、超参数搜索、SNN 量化、新 RTL、Vivado 工程、
FPGA 综合、实现、bitstream、功耗或板级验证；这些不是已完成的验证结果，后续是否执行
取决于阶段边界和用户明确任务。
