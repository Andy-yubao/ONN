# ONN — 光电神经网络项目

> **两套独立交付方向 · 4×4 器件采集 + 28×28 CNN FPGA 推理**

---

## 项目范围

本项目包含**两个独立交付方向**，当前不要求进行物理端到端连接：

### 主线 A：4×4 光学突触器件外围采集系统

为当前 4×4 光学突触阵列设计传感器外围电路，实现器件表征和数据采集：

```
受控光刺激 → 4×4 器件 → 模拟响应读出 → 信号调理与模数转换
→ 带时间戳的多通道数字曲线 → PC 保存、显示与后续特征分析
```

### 主线 B：28×28 CNN 的 FPGA 数字推理部署

在 Cyclone IV FPGA 上部署接受 28×28 手写数字输入的数字 CNN：

```
MNIST 图像 → Python 训练 → 量化/定点参考 → 权重导出
→ FPGA 定点 CNN 前向推理 → 0～9 分类结果
```

> 两部分当前不要求 A → B 端到端连接。详见 [docs/project_scope.md](docs/project_scope.md) 和 [docs/project_overview.md](docs/project_overview.md)。

---

## 当前硬件与工具

| 项目 | 说明 |
|------|------|
| 器件阵列 | 4×4 光学器件（16 单元），双列直插转接板 |
| 端口状态 | 数量、功能、电气规格 **均待确认** |
| FPGA 平台 | 小梅哥 AC620（Cyclone IV，具体型号待核实） |
| 开发工具 | Quartus II + ModelSim/Questa |
| 训练框架 | Python（PyTorch） |

> ⚠ 在获得官方 pinout 和安全电气规格前，**禁止通电**。详见 [device_bringup_checklist.md](docs/device_bringup_checklist.md)。

---

## 两条主线的关系

| 方面 | 说明 |
|------|------|
| 当前不要求 | 4×4 外围电路直接连接 FPGA CNN |
| 不构成 | 当前阶段的端到端实物系统 |
| 离线弱联系 | 器件曲线 → Python 模型参数更新 → 最终数字权重 → FPGA 部署 |
| FPGA 性质 | 仍执行标准数字 CNN 前向计算，物理器件不参与乘加 |
| 4×4 用途 | 传感器读出与器件表征，非 28×28 CNN 的输入 |

---

## 辅助研究：器件曲线优化器

器件曲线整理和自定义优化器探索属于**辅助研究**（不阻塞 A/B 交付）：

- 使用课题组论文数据建立器件曲线模型
- 复现差分电导权重更新方法（`W = G+ − G−`）
- 与普通 Adam 基线对照比较
- 普通 Adam CNN 是 FPGA 推理基线和对照，不是多余任务

详见 [docs/requirements.md](docs/requirements.md) 的 C 组需求和 [docs/development_plan.md](docs/development_plan.md) 的 C 轨计划。

---

## 当前阶段

> 项目总体范围已确定，进入双主线需求细化和原型设计阶段；器件安全接口和 CNN 具体架构仍待确认。

| 阶段 | 说明 | 状态 |
|------|------|------|
| **M0** | 课题定义、范围确认、文档统一 | ⚡ 进行中（文档重构） |
| **A 轨** | 4×4 采集系统设计 | 📋 待器件接口确认 |
| **B 轨** | 28×28 CNN FPGA 推理 | 📋 准备启动 |
| **C 轨** | 器件曲线辅助研究（不阻塞 A/B） | 📋 待启动 |

详见 [docs/development_plan.md](docs/development_plan.md)。

---

## 仓库目录与阅读顺序

```
ONN/
├── README.md                          ← 从这里开始
├── docs/
│   ├── project_scope.md               ← ⭐ 项目范围唯一基线（先读此）
│   ├── project_overview.md            ← 课题背景与双主线说明
│   ├── system_architecture.md         ← 系统架构（A/B/C 三部分）
│   ├── requirements.md                ← 分优先级需求
│   ├── interface_specification.md     ← 接口定义
│   ├── development_plan.md            ← 里程碑计划
│   ├── open_questions.md              ← 待确认问题
│   ├── device_bringup_checklist.md    ← ⚠ 上电安全清单
│   ├── decisions/                     ← 设计决策记录 (ADR)
│   └── meeting_notes/                 ← 会议记录
├── essay/
│   ├── README.md                      ← 论文目录与证据等级
│   ├── literature_review.md           ← 综合文献综述
│   ├── paper_matrix.md                ← 论文对比矩阵
│   └── paper_notes/                   ← 逐篇论文笔记
├── hardware/components/
│   └── optical_array.md               ← 4×4 光学阵列说明
├── fpga/                              ← FPGA 工程（RTL/仿真）
├── model/                             ← 训练模型
├── experiments/templates/
│   └── device_characterization.md     ← 实验记录模板
├── tools/                             ← 辅助工具脚本
└── THIRD_PARTY_NOTICES.md             ← 第三方材料许可说明
```

**推荐阅读顺序**：
1. `docs/project_scope.md` — 理解项目实际范围
2. `docs/project_overview.md` — 了解背景和双主线逻辑
3. `docs/system_architecture.md` — 看两张架构图和辅助关系
4. 其余文档按需查阅

---

## 安全提示

> ⚠ **重要**：在获得官方 pinout、端口拓扑和安全电气规格之前，禁止为 4×4 阵列连接电源、信号源或 FPGA GPIO。论文中的电压、电流、脉宽参数是候选参考，不构成当前封装阵列的 datasheet。详见 [docs/device_bringup_checklist.md](docs/device_bringup_checklist.md)。

---

## 第三方论文许可说明

本仓库 `essay/` 目录下的 PDF 论文**不**受根目录 MIT License 覆盖。每篇论文的版权归其出版方和作者所有。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

---

## 相关链接

- 文献综述：[essay/literature_review.md](essay/literature_review.md)
- 设计决策：[docs/decisions/](docs/decisions/)
