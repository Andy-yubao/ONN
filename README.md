# ONN 光电神经网络项目

## 项目简介

本项目研究面向低功耗、高速推理的光电神经网络（Optical Neural Network, ONN）系统，探索光学前端、模拟信号采集、神经网络模型与 FPGA 部署之间的协同设计。

利用二维半导体神经形态器件的电导特性参数，自定义适配器件特性的优化器与训练方法，在 PC 端训练轻量化网络，并在 FPGA 上完成前向推理部署，实现低功耗、高鲁棒性的类脑光电感知原型系统。

## 当前阶段

**M0：课题定义阶段**。正在阅读论文以明确系统输入形式、器件接口、网络结构和硬件方案。在 M0 完成前不启动具体实现工作。

## 可能的系统组成（待确认）

```
信号输入（形式待确认——可能有时间维度，非单帧静态图像）
    ↓
前端处理（是否需要外围电路待确认）
    ↓
FPGA 数据接收与处理
    ↓
神经网络前向推理（网络结构待确认）
    ↓
结果输出
```

> ⚠️ **注意**：目前系统的具体架构尚不明确。输入大概率不是由一瞬间的电信号决定，需要对一段时间内的数据做处理后才能送入网络。上述仅为可能的框架，将在 M0 阶段逐步明确。

- **神经形态器件（特性数据）**：提供电导态、STDP、LTP/LTD 等特性参数
- **外围电路**：是否需要及具体设计待确认
- **FPGA 平台**：小梅哥 AC620（Cyclone IV），Quartus II + ModelSim
- **神经网络**：训练框架 PyTorch（暂定），网络结构待确认

## 仓库目录说明

```
ONN/
├─ README.md              # 本文件——项目总览
├─ LICENSE                # 开源许可证
├─ .gitignore             # Git 忽略规则
├─ docs/                  # 项目文档
│  ├─ project_overview.md #   课题总览
│  ├─ requirements.md     #   需求文档
│  ├─ system_architecture.md  # 系统架构
│  ├─ interface_specification.md  # 接口定义
│  ├─ development_plan.md #   开发计划
│  ├─ open_questions.md   #   待确认问题
│  ├─ meeting_notes/      #   会议记录
│  └─ decisions/          #   设计决策记录
├─ hardware/              # 硬件设计（PCB、器件选型、datasheet）
├─ fpga/                  # FPGA 工程（RTL、仿真、约束、脚本）
├─ model/                 # 神经网络模型（训练、部署、评估）
├─ experiments/           # 实验记录
├─ tools/                 # 辅助工具脚本
└─ outputs/               # 输出产物（已 gitignore）
```

## 当前成员及分工

<!-- 待填写 -->

## 如何开始阅读项目

1. 从 `docs/project_overview.md` 了解课题背景和目标
2. 阅读 `docs/system_architecture.md` 了解系统总体架构
3. 阅读 `docs/development_plan.md` 了解当前开发阶段和计划
4. 查看 `docs/open_questions.md` 了解待确认问题

## 当前阶段的 main 任务（M0）

- [ ] 充分阅读论文，理解光学器件输出信号的空间和时间特性
- [ ] 明确系统的输入形式（信号 → 网络输入）的完整链路
- [ ] 明确网络结构（是否 CNN、输入张量形状、是否包含时间维度处理）
- [ ] 明确光学器件在系统中的职责
- [ ] 明确现有硬件条件（AC620 具体型号和资源）
- [ ] 明确暑假阶段目标
