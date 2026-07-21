# ONN — 光电神经网络项目

> **两条独立路线 · 光电突触阵列信号采集与光功率重建 + 类脑 CNN 训练研究与 FPGA 推理部署**

---

## 项目范围

本项目包含**两条在研究主题上相关、但在当前工程阶段不要求耦合的路线**。

### 路线 A：光电突触阵列信号采集与光功率重建

为课题组 `4×4` 光电突触阵列设计外围电路与 FPGA 数字处理系统，实现器件表征和光功率估计：

```
固定波长激光输入
  → 4×4 光电突触阵列
    → 偏置与电流读出（带偏置）
      → 模拟信号调理
        → ADC
          → FPGA 采样与缓存
            → 零点/增益校正
              → 响应归一化
                → 光功率映射
                  → 4×4 光功率矩阵
```

### 路线 B：类脑 CNN 训练研究与 FPGA 推理部署

以 MNIST `28×28` 单通道灰度图为输入，在 PC 端完成 CNN 训练和类脑学习规则探索，在 Cyclone IV FPGA 上部署定点前向推理：

```
MNIST 28×28 灰度图
  → PC 端基线训练（标准反向传播）
    → 类脑更新规则实验（候选方向）
      → 量化与参数导出
        → FPGA 定点 CNN 前向推理
          → 分类结果与性能评价
```

---

## 两条路线的关系

| 方面 | 说明 |
|------|------|
| 共同主题 | 低功耗类脑感知与计算研究 |
| 当前不要求 | 路线 A 输出直接输入路线 B（无强制数据接口） |
| 4×4 ≠ 28×28 输入 | 路线 A 输出 4×4 光功率矩阵，路线 B 输入 28×28 MNIST 图像 |
| 不构成 | 当前阶段的端到端实时分类系统 |
| FPGA 双重角色 | 路线 A：采样控制、缓存、校正、归一化、光功率计算；路线 B：CNN 前向推理 |
| 独立验证 | 两条路线可独立开发、独立验证和独立验收 |

详见 [docs/project_overview.md](docs/project_overview.md) 和 [docs/project_scope.md](docs/project_scope.md)。

---

## 当前硬件与工具

| 项目 | 说明 |
|------|------|
| 器件阵列 | 4×4 光电突触阵列（16 单元），当前尚无可用实物 |
| FPGA 平台 | 小梅哥 AC620（Cyclone IV，具体型号待核实） |
| 开发工具 | Quartus II + ModelSim/Questa |
| 训练框架 | Python（PyTorch），通过 Conda 管理环境 |

> ⚠ **安全提示**：在获得官方 pinout 和安全电气规格前，**禁止通电**。详见 [docs/device_bringup_checklist.md](docs/device_bringup_checklist.md)。

## 环境配置（模型训练）

本项目使用 Conda 管理模型训练环境。环境配置文件位于项目根目录：

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate onn

# 更新环境（依赖变更后）
conda env update -f environment.yml
```

详见 [`environment.yml`](environment.yml) 和 [`model/README.md`](model/README.md)。

---

## 当前阶段

> 项目架构与边界定义已在架构层面完成。器件参数和工程选型仍有未知项，但已具有明确的实验验证路径，不阻塞下一阶段工作。

| 阶段 | 说明 | 状态 |
|------|------|:----:|
| **M0** | 项目架构与边界定义 | ✅ 架构层面已完成 |
| **M1-A** | 器件测量方案和前端采集原型 | 📋 待启动 |
| **M1-B** | MNIST 软件基线与 Tiny-ResNet 基线 | ✅ **已完成** |

详见 [docs/development_plan.md](docs/development_plan.md)。

---

## 仓库目录与阅读顺序

```
ONN/
├── README.md                          ← 从这里开始
├── environment.yml                    ← 🌿 Conda 环境配置（模型训练）
├── docs/
│   ├── project_overview.md            ← ⭐ 双路线背景与目标（先读此）
│   ├── project_scope.md               ← 项目范围基线
│   ├── system_architecture.md         ← 两张独立架构图
│   ├── requirements.md                ← 分路线分组需求
│   ├── interface_specification.md     ← 数据接口定义
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
├── model/                             ← 训练模型（Conda 环境）
├── experiments/templates/
│   └── device_characterization.md     ← 实验记录模板
├── tools/                             ← 辅助工具脚本
└── THIRD_PARTY_NOTICES.md             ← 第三方材料许可说明
```

**推荐阅读顺序**：
1. `docs/project_overview.md` — 了解双路线背景和目标
2. `docs/system_architecture.md` — 看两张独立架构图
3. 其余文档按需查阅

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
