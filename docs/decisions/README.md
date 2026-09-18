# 设计决策记录

> 每当遇到一个重要选择，在此记录决策过程。
> 文件命名方式：`NNN-decision-title.md`
>
> 这里的 ADR 保留历史决策链。当前 active scope、架构、阶段和开放问题分别以
> [`project_scope.md`](../project_scope.md)、[`system_architecture.md`](../system_architecture.md)、
> [`development_plan.md`](../development_plan.md) 和 [`open_questions.md`](../open_questions.md)
> 为准；旧 28×28 CNN / AC620 决策不代表当前部署目标。

## 记录原则

每份决策记录包括：

1. **问题**：需要决策的问题是什么
2. **方案**：考虑过哪些选项
3. **选择**：最终采用了哪个方案
4. **原因**：选择该方案的理由
5. **后果**：该选择可能带来的影响

## 已记录的决策

- [001-当前硬件边界为 4×4](001-current-hardware-scope-is-4x4.md)
- [002-第一版数字基线网络选择](002-baseline-network-selection.md)
- [003-项目有两个独立交付方向](003-two-independent-deliverables.md)
- [004-当前不要求端到端集成](004-no-current-end-to-end-integration.md)
- [005-外围采集系统支持多档可配置采样率](005-configurable-multirate-acquisition.md)
- [006-FPGA 实现 28×28 数字 CNN 推理](006-28x28-digital-cnn-on-fpga.md)
- [007-双路线架构与边界定义](007-dual-track-architecture-and-boundaries.md)
- [008-器件模型驱动训练与多次采样采集](008-device-model-driven-training-and-temporal-acquisition.md)
