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

### stem 卷积引擎流程（stem_conv_serial，2026-08-01 落地）

- 第一版单 MAC 串行 stem 卷积引擎：`rtl/stem_conv_serial.v`
  （实现 `input_q → stem Conv2d(1→16,3×3,pad1) → +bias → INT32 饱和 → requantize_u8 → stem_q`；
  冻结设计见 `docs/rtl_microarchitecture.md`）
- 同步存储模板：`rtl/sync_ram_u8.v`、`rtl/sync_rom_s8.v`、`rtl/sync_rom_s32.v`
  （一拍延迟同步读，无厂商 IP，Quartus 推断 M9K）
- Quartus smoke 工程：`stem_conv_smoke`（`quartus/stem_conv_smoke.{qpf,qsf}` 提交）；
  由 `scripts/create_stem_conv_project.tcl` 创建，`run_quartus_smoke.ps1
  -ProjectName stem_conv_smoke -CreateScript create_stem_conv_project.tcl` 编译
- Questa 验证：`tb/tb_stem_conv_serial.v`（digit8 黄金，conv1_acc/stem_q 流/读回 12544×3 逐位一致）
  + `tb/tb_stem_conv_padding.v`（padding 专项：四角 4/边缘 6/中心 9 有效 tap），已并入 `run_questa.ps1`
- Python 契约测试：`model/tests/test_stem_rtl_contract.py`（地址公式/元素数/tap 数/mult 常数/signed 解释），已并入 `run_all.ps1`
- `run_all.ps1` 当时为 7 步：工具链 → 器件 → Questa（requant/GAP/stem/padding）→ 算术 smoke → stem smoke → Python 契约 → stem 契约

### 流式 MaxPool 流程（maxpool2x2_stream，2026-08-01 落地）

- 独立可综合流式 `2×2 stride=2 MaxPool` 原始模块：`rtl/maxpool2x2_stream.v`
  （直接消费 `stem_q` 流 → `pool1_q`；只存上一条偶数行 `row_buffer` 28×8；无除法/取模/乘法，
  UINT8 比较树；oc/y/x 计数仅在 `busy && in_valid` 前进；冻结设计见 `docs/rtl_microarchitecture.md` §9）
- Questa 验证：`tb/tb_maxpool2x2_stream.v` 两遍 golden——连续输入 + 每 5 发插 2 空拍，
  均 3136/3136 逐位一致、空拍冻结检查、专项手工重算 5 窗口、无 X/Z，已并入 `run_questa.ps1`
- Quartus smoke 工程：`maxpool_smoke`（`quartus/maxpool_smoke.{qpf,qsf}` 提交）；
  由 `scripts/create_maxpool_project.tcl` 创建，`run_quartus_smoke.ps1
  -ProjectName maxpool_smoke -CreateScript create_maxpool_project.tcl` 编译（纯逻辑，无 M9K/乘法器/除法器）
- Python 契约测试：`model/tests/test_maxpool_rtl_contract.py`（元素数/CHW 池地址映射/窗口抽查/UINT8 范围/全量 3136 重算），已并入 `run_all.ps1`
- `run_all.ps1` 现为 10 步：工具链 → 器件 → Questa（requant/GAP/stem/padding/maxpool/集成）→ 算术 smoke → stem smoke → Python 契约 → stem 契约 → maxpool smoke → maxpool 契约 → 集成 smoke
- stem 输出 RAM 通过 `STORE_OUTPUT_RAM` 参数控制（默认 1 保留历史回归；集成工程用 0 不实例化）

### 集成流水线流程（stem_pool1_pipeline，2026-08-01 落地）

- 集成核心：`rtl/stem_pool1_pipeline.v`，实现 `input_q RAM → stem_conv_serial
  (STORE_OUTPUT_RAM=0) → 流式 MaxPool → pool1 RAM（3136×UINT8）`；冻结设计见
  `docs/rtl_microarchitecture.md` §10
- **同一个 `start` 同时启动 stem 与 MaxPool**；`stem_done` 不用于启动 MaxPool
  （它发生在全部 stem_q 发送之后）；`start` 与第一项 `in_valid` 不在同一采样沿
- **`busy = stem_busy || maxpool_busy` 是状态信号、非背压**；无 ready/stall，
  stem 不被 MaxPool 暂停；`done = maxpool_done`（实测与 stem_done 同周期）
- `stem_conv_serial` 新增 `STORE_OUTPUT_RAM` 参数（Verilog generate）：默认 1
  保留完整 12544×8 输出 RAM（原 testbench / `stem_conv_smoke` 回归不变）；0 时
  不实例化、`output_rdata` 固定 0、保留 `q_valid` 流
- Questa 验证：`tb/tb_stem_pool1_pipeline.v`（digit8，两遍推理逐位一致，第二遍
  reset 后重跑结果相同）：stem_q/conv1_acc 流 12544/12544、pool1 流 3136/3136、
  pool1 RAM 读回 3136/3136；`stem_q_valid ⇒ maxpool_busy`、pool 地址连续 0..3135、
  done 前恰好 12544 个 stem_q 与 3136 个 pool1_q、busy 期间额外 start 被忽略、
  无 X/Z，已并入 `run_questa.ps1`
- Quartus smoke 工程：`stem_pool1_smoke`（`quartus/stem_pool1_smoke.{qpf,qsf}`
  提交）；由 `scripts/create_stem_pool1_project.tcl` 创建，`run_quartus_smoke.ps1
  -ProjectName stem_pool1_smoke -CreateScript create_stem_pool1_project.tcl` 编译
- 集成工程 M9K 分项（Fitter 实测）：input RAM 1 + weight ROM 1 + **pool1 RAM 4**
  + bias 落入逻辑（12 LC/12 reg）；**完整 stem 输出 RAM 已从集成工程消失**
  （Memory bits 32,512 = 6272 + 1152 + 25088）

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
