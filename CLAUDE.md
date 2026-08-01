# ONN 项目

光学神经网络（ONN）FPGA 部署研究项目。GitHub：https://github.com/Andy-yubao/ONN

模型代码位于 `model/`（独立可交付模块）；`experiments/` 存放实验记录与结果。

## 会话约定（必须遵守）

- **用中文回答问题**：所有回复一律使用中文。

## 开发环境（必须遵守）

**必须使用 `onn` conda 环境** —— base/默认 anaconda 环境没有 torch。

- 环境名：`onn`，路径：`D:\tools\anaconda3\envs\onn`
  - PowerShell：`conda activate onn`
  - Git Bash 直接调用：`/d/tools/anaconda3/envs/onn/python.exe <script>`
- torch 2.11.0+cu128（CUDA 可用）
- 依赖清单：`model/requirements.txt`（torch>=2.0、torchvision>=0.15、numpy>=1.24、matplotlib>=3.7、pytest>=7.0）
- 仓库内没有 venv，一律使用 conda env `onn`

### 运行约定
- 所有命令从**仓库根目录**运行，例如：
  `python model/train.py --config model/configs/baseline_cnn.json`
- `onn_model` 包通过脚本所在目录（`model/`）自动加入 sys.path，无需手动设置 PYTHONPATH
- 常用入口：`model/train.py`、`model/evaluate.py`、`model/profile_model.py`、`model/analyze_activations.py`、`model/scripts/run_m1_baselines.ps1`、`model/scripts/run_m2_compact_models.ps1`
- 测试：`python -m pytest model/tests`
- MNIST 首次下载需代理：`set HTTP_PROXY=http://127.0.0.1:10809`

## 关键目录

| 路径 | 说明 | git |
|---|---|---|
| `model/onn_model/` | 核心包（模型、引擎、数据、量化、profile） | 跟踪 |
| `model/configs/` | 训练配置 JSON（baseline_cnn、tiny_resnet、micro_cnn_*） | 跟踪 |
| `model/runs/` | 训练输出：`.pt` checkpoint、history.csv、metrics.json | **gitignored，权重仅本地** |
| `model/data/` | MNIST 数据 | 忽略 |
| `experiments/` | 实验记录与结果（含 multiseed） | 跟踪 |
| `fpga/baseline_cnn/` | FPGA 参数包、黄金向量、scripts（`params/`、`sim/vectors/`、`scripts/`） | 跟踪 |

## FPGA 工具链（BaselineCNN）

- 工具链探测脚本：`fpga/baseline_cnn/scripts/check_toolchain.ps1`（无 GUI、不改系统 PATH）
  - PowerShell 运行：`.\fpga\baseline_cnn\scripts\check_toolchain.ps1`
  - 优先读取环境变量 `QUARTUS_BIN` / `QUESTA_BIN`（指向 bin 目录或直接指向 .exe），未设置时使用下列默认路径
  - 任一工具缺失或版本命令失败时以非零退出码结束
- 已确认安装（2026-08-01 探测通过）：
  - **Quartus Prime Shell 25.1std.0 Build 1129（Lite）**
    `D:\tools\altera_lite\25.1std\quartus\bin64\quartus_sh.exe`
  - **Questa Altera Starter FPGA Edition 2025.2**（vlog / vsim 同目录）
    `D:\tools\altera_lite\25.1std\questa_fse\win64\vlog.exe`
    `D:\tools\altera_lite\25.1std\questa_fse\win64\vsim.exe`

### 算术 smoke 流程（requantize_u8 / gap_div49，2026-08-01 落地）

- 统一入口：`fpga/baseline_cnn/scripts/run_all.ps1`
  （工具链探测 → 器件检查 → Questa 黄金向量 → Quartus smoke 综合 → Python 契约测试）
- 分步：`run_questa.ps1`（requant 20384 项 + GAP 穷举/黄金向量）、`run_quartus_smoke.ps1`
  （device 检查 → 建工程 → compile → 资源/警告报告）
- 工程与 RTL：`fpga/baseline_cnn/quartus/`（`.qpf`/`.qsf` 提交，缓存忽略）、`fpga/baseline_cnn/rtl/`
- 顶层 `baseline_cnn_smoke_top` 为寄存器包装 + 全部 VIRTUAL_PIN（Fitter 拒绝纯组合虚拟引脚，
  Error 171016），非最终板级顶层

## FPGA 硬件目标与开发约定（BaselineCNN）

- **目标开发板**：小梅哥 AC620
- **目标 FPGA**：EP4CE10F17C8（器件系列：Cyclone IV E；封装：F17；速度等级：C8）
- **器件已冻结**：所有 Quartus 工程、综合报告和资源预算都必须针对该器件；**不得**为编译方便临时切换其他器件
- 板级引脚**尚未冻结**：AC620 的主时钟、UART、复位、调试 LED 引脚均未确定
- **未获得可靠板卡引脚表前，不得编造引脚约束**
- 硬件目标明细见 `fpga/baseline_cnn/docs/hardware_target.md`

## 模型与分支基线

- 模型定义在 `model/onn_model/models/`：
  - **M1 交付候选**：`BaselineCNN`（第一 FPGA 部署候选）、`TinyResNet`（高准确率备选 / 软件参考）
  - **M2 compact**：`MicroCNNSmall` / `MicroCNNExtraSmall` / `DepthwiseMicroCNN`（轻量化探索）
- 当前模型开发基线分支：`model/m1-baseline-audit`；**不要从 `master` 重新开始**
- 历史权重（`model/runs/` 本地 .pt）已验证可用当前模型定义以 strict=True 加载
