# ONN 项目 AI 开发代理指令

> 指令内容维护于 `docs/ai_agent_instructions.md`；`AGENTS.md` 与 `CLAUDE.md` 由
> `tools/sync_ai_agent_instructions.py` 生成并保持**字节级一致**，**请勿直接修改两个根文件**。
> 修改指令后运行 `python tools/sync_ai_agent_instructions.py`（重新生成）或
> `python tools/sync_ai_agent_instructions.py --check`（只检查一致性，漂移时非零退出）。

---

# ONN 项目

光学神经网络（ONN）FPGA 部署研究项目。GitHub：https://github.com/Andy-yubao/ONN

模型代码位于 `model/`（独立可交付模块）；`experiments/` 存放实验记录与结果。

## 会话约定（必须遵守）

- **用中文回答问题**：所有回复一律使用中文。
- **不得虚构验证**：没有实际运行的命令、测试或流程，不得声称"通过"；最终报告必须如实写明"未运行"。
- **不要从旧 `master` 重新开始**：远程默认分支 `master` 明显落后，开发以活跃功能分支的最新 HEAD 为基线。

## 开发环境（必须遵守）

**必须使用 `onn` conda 环境** —— base/默认 anaconda 环境没有 torch。

- 环境名：`onn`，路径：`D:\tools\anaconda3\envs\onn`
  - PowerShell：`conda activate onn`
  - Git Bash 直接调用：`/d/tools/anaconda3/envs/onn/python.exe <script>`
- torch 2.11.0+cu128（CUDA 可用）；依赖清单见 `model/requirements.txt`
- 仓库内没有 venv，一律使用 conda env `onn`

### 运行约定
- 所有命令从**仓库根目录**运行，例如：`python model/train.py --config model/configs/baseline_cnn.json`
- `onn_model` 包通过脚本所在目录（`model/`）自动加入 sys.path
- 常用入口：`model/train.py`、`model/evaluate.py`、`model/profile_model.py`、`model/analyze_activations.py`、`model/export_baseline_cnn_hardware.py`、`model/verify_*.py`、`model/scripts/run_m1_baselines.ps1`、`model/scripts/run_m2_compact_models.ps1`
- 测试：`python -m pytest model/tests`
- MNIST 首次下载需代理：`set HTTP_PROXY=http://127.0.0.1:10809`

## 关键目录

| 路径 | 说明 | git |
|---|---|---|
| `model/onn_model/` | 核心包（模型、引擎、数据、量化、profile、int8 参考） | 跟踪 |
| `model/configs/` | 训练配置 JSON（baseline_cnn、tiny_resnet、micro_cnn_*） | 跟踪 |
| `model/runs/` | 训练输出：`.pt` checkpoint、history.csv、metrics.json | **gitignored，权重仅本地** |
| `model/data/` | MNIST 数据 | 忽略 |
| `experiments/` | 实验记录与结果（model_baselines / compaction / deployment） | 跟踪 |
| `fpga/baseline_cnn/` | FPGA 参数包、RTL、TB、Quartus 工程、黄金向量、docs | 跟踪 |

## FPGA 工具链（BaselineCNN）

- 工具链探测脚本：`fpga/baseline_cnn/scripts/check_toolchain.ps1`（无 GUI、不改系统 PATH）
- 已确认安装：
  - **Quartus Prime Shell 25.1std.0 Build 1129（Lite）**：`D:\tools\altera_lite\25.1std\quartus\bin64\quartus_sh.exe`
  - **Questa Altera Starter FPGA Edition 2025.2**：`D:\tools\altera_lite\25.1std\questa_fse\win64\vlog.exe` / `vsim.exe`

### FPGA 验证命令（常用）
- 完整回归入口：`.\fpga\baseline_cnn\scripts\run_all.ps1`（工具链 → 器件 → Questa → Quartus smoke → Python 契约）
- 单独 Questa：`.\fpga\baseline_cnn\scripts\run_questa.ps1`
- 单独 Quartus smoke：`.\fpga\baseline_cnn\scripts\run_quartus_smoke.ps1 -ProjectName <name> -CreateScript <create.tcl>`

## FPGA 硬件目标与当前部署边界（BaselineCNN）

- **目标开发板**：小梅哥 AC620 V2；**目标 FPGA**：EP4CE10F17C8（Cyclone IV E / F17 / C8）；器件已冻结
- **目标时钟**：50 MHz；**当前最小 Fmax**：56.41 MHz；最小 setup slack **+2.274 ns**（Slow 85C）
- **完整核心固定周期**：1,590,315 cycles ≈ 31.81 ms @ 50 MHz
- **固定 digit8 板级自检已完成**：固定 digit8、片上 ROM 输入、JTAG SRAM 配置、prediction=8、PASS LED；本次**未写 EPCS Flash**，断电后配置消失
- **板级验证边界（不得扩大）**：固定 digit8 片上 ROM 自检通过 ≠ 任意输入部署。外部图像输入、UART 图像传输、测试集级硬件准确率（≥1000 张）、功耗实测、EPCS Flash 固化均**未完成**
- **引脚状态**：主时钟（PIN_E1，50 MHz）与 4 颗 LED（A2/B3/A4/A3）已用于正式板级验证并视为已确认；UART、复位、按键引脚**尚未冻结**
- **不得编造引脚约束**：未获得可靠板卡引脚表前，不得猜测引脚
- 硬件目标明细：`fpga/baseline_cnn/docs/hardware_target.md`

## 文档事实源优先级

发生冲突时按以下优先级审计，**不得凭"哪个文档写得更像真的"选择**：

1. **可执行实现与冻结配置**：`model/onn_model/`、`model/configs/`、`model/export_baseline_cnn_hardware.py`、`model/verify_*.py`、`model/tests/`、`fpga/baseline_cnn/{rtl,tb,scripts,quartus,params,sim/vectors}/`
2. **实验与硬件验证证据**：`experiments/**/README.md` 与结果文件、`fpga/baseline_cnn/docs/{data_format,hardware_target,rtl_microarchitecture}.md`、`docs/model_architecture.md`
3. **最新项目决策与会议记录**：`docs/meeting_notes/`、`docs/decisions/`、`docs/open_questions.md`
4. **项目级汇总文档（需被校正，不得作为唯一证据）**：`README.md`、`docs/project_scope.md` 等

每类事实只设一个主要位置，其他文档只做摘要并链接：

| 事实类型 | 主要文档 |
|---|---|
| 项目当前范围与状态 | `docs/project_scope.md` |
| 设计决策历史 | `docs/decisions/` |
| 模型结构 | `docs/model_architecture.md` |
| 量化数据格式 | `fpga/baseline_cnn/docs/data_format.md` |
| FPGA 目标、资源、STA、板级状态 | `fpga/baseline_cnn/docs/hardware_target.md` |
| RTL 协议与微架构 | `fpga/baseline_cnn/docs/rtl_microarchitecture.md` |
| 实验数值 | `experiments/**/README.md` 与结果文件 |

## 文档修改约定

- 修改 `docs/ai_agent_instructions.md` 后运行 `python tools/sync_ai_agent_instructions.py` 重新生成 `AGENTS.md` 与 `CLAUDE.md`
- 修改文档后运行检查：`python tools/sync_ai_agent_instructions.py --check`（AGENTS/CLAUDE 一致性）与 `python tools/check_markdown_links.py`（相对链接）
- `git diff --check` 避免尾随空格与冲突标记
- 数值（周期、资源、准确率、周期数）从主文档/实验记录读取，不得凭记忆填写

## 模型与分支基线

- 模型定义在 `model/onn_model/models/`：`BaselineCNN`（FPGA 部署模型）、`TinyResNet`（软件参考）、`MicroCNNSmall`/`MicroCNNExtraSmall`/`DepthwiseMicroCNN`（M2 compact）
- **不要从旧 `master` 重新开始**；开发以活跃功能分支的最新 HEAD 为基线
- 历史权重（`model/runs/` 本地 .pt）已验证可用当前模型定义以 strict=True 加载
