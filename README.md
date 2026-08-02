# ONN — 光学神经网络项目

> **两条研究路线 · 光电突触阵列表征与多次采样采集 + 器件模型驱动训练与 FPGA 数字推理部署**

---

## 项目范围

本项目包含**两条在研究主题上相关、在当前工程阶段不要求实时耦合、但允许并需要离线算法耦合的路线**。

### 路线 A：光电突触阵列信号采集与光功率重建

为课题组 `4×4` 光电突触阵列设计外围电路与 FPGA 数字处理系统。**原始交付物是带时间戳的多通道电流/电压/ADC 响应序列**，支持多次采样和动态曲线记录；光功率估计仅在明确测量协议下成立（固定波长、偏置、初始状态或恢复条件、光照历史、等待时间、采样时刻或统计窗口、温度等）。`4×4` 光功率矩阵是受条件约束的目标输出，不是无条件唯一反演，也不覆盖原始时间序列。

```
固定波长激光输入
  → 4×4 光电突触阵列
    → 偏置与电流读出（带偏置）
      → 模拟信号调理
        → ADC
          → FPGA 采样与缓存（多次采样 / 动态曲线）
            → 零点/增益校正
              → 归一化
                → 协议限定下的光功率估计 → 4×4 光功率矩阵（受条件约束）
```

### 路线 B：器件模型驱动训练研究与 FPGA 数字推理部署

**已完成的工程基线**：以 MNIST `28×28` 单通道灰度图为输入的 BaselineCNN / Tiny-ResNet 软件对照基线，以及 BaselineCNN INT8 定点前向的完整数字硬件实现（RTL、仿真、50 MHz STA、固定 digit8 板级自检）。

**当前新增研究方向**（2026-08-02 组会）：将标准 RGB 或其他图像经器件响应模型转换为电导/电流表示后训练与评价；定向 Hebbian 连接增强；探索 KAN；考虑器件的持续特性与多次采样。

```
（已完成）MNIST 28×28 → PC 基线训练 → BN 融合 → INT8 PTQ → 整数参考模型
  → 参数导出 → FPGA 定点 CNN 前向推理 → 完整 RTL → 50 MHz STA → 固定 digit8 板级自检
（新增研究）标准 RGB 图像 → 器件响应模型 → 电导/电流表示 → 训练与评价（Hebbian / KAN / 多次采样）
```

详见 [docs/project_scope.md](docs/project_scope.md)、[docs/project_overview.md](docs/project_overview.md) 与 [docs/decisions/008-device-model-driven-training-and-temporal-acquisition.md](docs/decisions/008-device-model-driven-training-and-temporal-acquisition.md)。

---

## 两条路线的关系

| 方面 | 说明 |
|------|------|
| 共同主题 | 低功耗类脑感知与计算研究 |
| 已完成基线 | MNIST BaselineCNN/Tiny-ResNet 软件基线与 BaselineCNN INT8 FPGA 数字部署，均保留为已完成工作 |
| 当前不要求 | 实时硬件端到端联调（路线 A 输出直接实时输入路线 B 不强制） |
| 允许且需要 | **离线算法耦合**：路线 A 的器件表征和响应模型可为路线 B 的训练数据变换提供依据 |
| 4×4 ≠ 28×28 | 路线 A 输出 4×4 光功率矩阵，路线 B 输入 28×28 MNIST 图像 |
| FPGA 双重角色 | 路线 A：采样控制、缓存、校正、归一化、协议限定光功率估计；路线 B：CNN 前向推理 |
| 独立验证 | 两条路线可独立开发、独立验证和独立验收 |

---

## 当前硬件与工具

| 项目 | 说明 |
|------|------|
| 器件阵列 | 4×4 光电突触阵列（16 单元），当前尚无可用实物；持续特性（状态依赖、恢复条件、多次采样）尚未实测 |
| FPGA 平台 | 小梅哥 AC620 V2（Cyclone IV E，EP4CE10F17C8，已冻结），目标时钟 50 MHz |
| FPGA 引脚 | 主时钟（PIN_E1）与 4 个 LED（A2/B3/A4/A3）已用于正式板级验证并视为已确认；UART、复位、按键等接口仍待冻结 |
| 开发工具 | Quartus Prime Lite 25.1std + Questa 2025.2（不使用 Vivado / FINN / PYNQ / Xilinx 工具链） |
| 训练框架 | Python（PyTorch 2.11.0+cu128），通过 `onn` Conda 环境管理 |

> ⚠ **安全提示**：在获得官方 pinout 和安全电气规格前，**禁止为 4×4 阵列通电**。详见 [docs/device_bringup_checklist.md](docs/device_bringup_checklist.md)。
> ⚠ 会议中"约 20 秒到 1 分钟达到稳态"是 **2026-08-02 组会的初步估计**，必须等待实物验证，不构成 datasheet 参数。

## 当前进度

> **已完成的工程基线**：MNIST 浮点基线（BaselineCNN / Tiny-ResNet）、基线审计与可复现性加固、BaselineCNN BN 融合、INT8 PTQ、纯整数参考模型、定点参数导出、完整 BaselineCNN 纯计算核心 RTL、模块级与集成级仿真验证、Quartus 综合与资源报告、50 MHz STA、以及**固定 digit8 片上 ROM 输入的 AC620 板级自检**。

| 里程碑 | 说明 | 状态 |
|------|------|:----:|
| **M0** | 项目架构与边界定义 | ✅ 已完成 |
| **M1-A** | 器件测量方案和前端采集原型 | 📋 待启动 |
| **M1-B** | MNIST 软件基线（BaselineCNN / Tiny-ResNet） | ✅ 已完成 |
| **M1-B.1** | 基线实验审计与可复现性加固 | ✅ 已完成 |
| **M3-B.1** | BN 融合、INT8 PTQ、整数参考与参数导出 | ✅ 已完成 |
| **M3-B.2** | 完整 BaselineCNN RTL 与仿真验证 | ✅ 已完成 |
| **M3-B.3** | 50 MHz STA 与固定 digit8 板级自检 | ✅ 已完成 |
| **M3-B.4** | 外部输入与测试集级硬件评价 | ⬜ 未完成 |
| **M2-B** | 器件模型驱动训练 / Hebbian / KAN | 🔬 研究中 / 待启动 |
| **M4-B** | 功耗、性能和最终报告 | ⬜ 未完成 |

> **当前板级验证边界**：固定 digit8、片上 ROM 输入的完整 BaselineCNN 计算核心已在 AC620 V2 上完成 50 MHz JTAG SRAM 板级自检；**外部图像输入、UART 图像传输、测试集级硬件准确率、功耗测量和 EPCS Flash 固化尚未完成**。详见 [docs/development_plan.md](docs/development_plan.md) 与 [fpga/baseline_cnn/docs/hardware_target.md](fpga/baseline_cnn/docs/hardware_target.md)。

---

## 仓库目录与阅读顺序

```
ONN/
├── README.md                          ← 从这里开始
├── environment.yml                    ← Conda 环境配置（模型训练）
├── AGENTS.md / CLAUDE.md              ← AI 开发代理指令（由 docs/ai_agent_instructions.md 自动生成，勿直接修改）
├── docs/
│   ├── project_overview.md            ← 双路线背景与目标（先读此）
│   ├── project_scope.md               ← 项目范围与当前状态基线
│   ├── system_architecture.md         ← 架构关系
│   ├── requirements.md                ← 需求与状态
│   ├── interface_specification.md     ← 数据接口定义
│   ├── development_plan.md            ← 里程碑计划（M1-B/M3-B.x 状态）
│   ├── open_questions.md              ← 待确认 / 开放问题
│   ├── device_bringup_checklist.md    ← ⚠ 上电安全清单
│   ├── ai_agent_instructions.md       ← AI 开发代理指令唯一可编辑源
│   ├── decisions/                     ← 设计决策记录 (ADR)
│   └── meeting_notes/                 ← 会议记录
├── essay/                             ← 论文目录与证据等级
├── hardware/components/
│   └── optical_array.md               ← 4×4 光学阵列说明
├── fpga/
│   └── baseline_cnn/                  ← 参数包、RTL、TB、Quartus 工程、docs
│       └── docs/
│           ├── data_format.md         ← 量化数据格式（唯一数值标准）
│           ├── hardware_target.md     ← FPGA 目标、资源、STA、板级状态
│           └── rtl_microarchitecture.md ← RTL 协议与微架构（冻结设计）
├── model/                             ← 训练模型（Conda 环境）
├── experiments/                       ← 实验记录与结果（model_baselines/compaction/deployment）
├── tools/                             ← 辅助工具与文档检查脚本
└── THIRD_PARTY_NOTICES.md             ← 第三方材料许可说明
```

**推荐阅读顺序**：
1. `docs/project_overview.md` — 双路线背景和目标
2. `docs/project_scope.md` — 项目范围与当前状态
3. `fpga/baseline_cnn/docs/hardware_target.md` — FPGA 硬件目标与板级验证边界

---

## 安全提示

> ⚠ **重要**：在获得官方 pinout、端口拓扑和安全电气规格之前，禁止为 4×4 阵列连接电源、信号源或 FPGA GPIO。论文中的电压、电流、脉宽参数是候选参考，不构成当前封装阵列的 datasheet。会议估计的稳态时间（约 20 秒到 1 分钟）同样**尚未实测**。详见 [docs/device_bringup_checklist.md](docs/device_bringup_checklist.md)。

---

## 第三方论文许可说明

本仓库 `essay/` 目录下的 PDF 论文**不**受根目录 MIT License 覆盖。每篇论文的版权归其出版方和作者所有。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

---

## 相关链接

- 文献综述：[essay/literature_review.md](essay/literature_review.md)
- 设计决策：[docs/decisions/](docs/decisions/)
- 模型说明：[model/README.md](model/README.md)
- 硬件目标：[fpga/baseline_cnn/docs/hardware_target.md](fpga/baseline_cnn/docs/hardware_target.md)
