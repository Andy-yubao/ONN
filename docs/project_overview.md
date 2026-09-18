# 项目总览

ONN 当前是一个器件动态模型驱动的 8×8 脉冲神经网络研究与 FPGA 部署项目。主线从
统一的 first-spike latency encoding 开始，先完成参数量相近的 CNN/SNN 公平比较，再
选择是否将 SNN promote 为正式部署模型。

## 研究链路

```text
器件动态模型 → 8×8 first-spike 编码 → CNN/SNN 候选 → 公平比较
→ champion 选择 → 量化/整数参考 → Basys3 / Artix-7 FPGA
```

当前没有 production SNN，也没有当前 SNN 的量化、RTL 或板级结果。旧 28×28 CNN / AC620
成果属于已冻结的 legacy 阶段，完整历史入口见 [`history/README.md`](../history/README.md)。

## 当前资产

- 三项已记录的 8×8 SNN 实验在 [`experiments/snn/`](../experiments/snn/)；
- 共享编码器和轻量测试在 [`experiments/common/`](../experiments/common/)；
- CNN 计划在 [`experiments/cnn/README.md`](../experiments/cnn/README.md)；
- 公平比较协议在 [`experiments/comparison/README.md`](../experiments/comparison/README.md)。

当前范围、架构、阶段计划和开放问题分别以 [`project_scope.md`](project_scope.md)、
[`system_architecture.md`](system_architecture.md)、[`development_plan.md`](development_plan.md)
和 [`open_questions.md`](open_questions.md) 为唯一来源。
