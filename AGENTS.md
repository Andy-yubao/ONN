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
- `run_all.ps1` 现为 12 步：工具链 → 器件 → Questa（requant/GAP/stem/padding/maxpool/集成/conv2-pool2/conv3）→ 算术 smoke → stem smoke → Python 契约 → stem 契约 → maxpool smoke → maxpool 契约 → 集成 smoke → conv23 smoke → conv23 契约
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

### 共享 conv2/conv3 引擎流程（conv_u8_serial，2026-08-01 落地）

- 共享单 MAC 卷积引擎：`rtl/conv_u8_serial.v`，**只服务 conv2/conv3**（不替换
  stem）；`layer_sel`（0=conv2，1=conv3）在 `start` 时锁存，运行期改变不影响当前
  推理；输入特征图 RAM **在模块外部**（引擎驱动 `fm_raddr`，外部同步 RAM 一拍
  延迟返回 `fm_rdata`）；内部实例化 conv2/conv3 权重 + bias 共 4 个 ROM；冻结设计
  见 `docs/rtl_microarchitecture.md` §11
- 遍历 `oc→y→x→ic→ky→kx`，输出 CHW；**不用除法/取模**（直接维护 ic/ky/kx 嵌套
  计数器，地址常数乘法全用移位/加减：196=128+64+4、49=32+16+1、144=128+16、
  288=256+32、14=16−2、7=8−1、9=8+1、3=2+1）；padding 越界贡献 0 且不生成非法
  RAM 地址；乘积 ≥17 位有符号、精确累加 signed 64-bit、bias 后 INT32 饱和、
  复用 `requantize_u8.v`（multiplier 按 layer 选择：conv2 1097020857 / conv3
  1298974956，shift 38）
- `maxpool2x2_stream` 已**参数化**支持 pool2：`N_CH/IN_H/IN_W/OC_W/XY_W/
  OUT_ADDR_W`，默认=stem 配置原测试不变；pool2 配置 `N_CH=32, IN_H=14, IN_W=14,
  OC_W=5, XY_W=4, OUT_ADDR_W=11`
- **busy/done 协议（冻结，本阶段修复）**：stem 与 maxpool 的 `busy` 均覆盖
  `S_DONE`——从 start 起保持 1 直到 done 脉冲，无 `busy=0 && done=0` 窗口；
  done 高电平期间 busy 可为 0；`busy_cycles == done 周期 − start 周期`；
  连续两次启动（done 后立即再 start，无 reset）不丢失
- Questa 验证：`tb/tb_conv2_pool2.v`（digit8：conv2 与 pool2 同一 start 同启，
  conv2_acc/conv2_q 6272/6272 + pool2_q 1568/1568 逐位一致、`q_valid ⇒ pool_busy`、
  conv_done 与 pool_done 同周期、非法 fm_raddr 0 次、无 X/Z）+
  `tb/tb_conv3_serial.v`（conv3_acc/conv3_q 1568/1568 两遍一致、reset 后重跑逐位
  相同、非法 fm_raddr 0 次），已并入 `run_questa.ps1`
- Quartus smoke 工程：`conv23_smoke`（`quartus/conv23_smoke.{qpf,qsf}` 提交）；
  由 `scripts/create_conv23_project.tcl` 创建，`run_quartus_smoke.ps1
  -ProjectName conv23_smoke -CreateScript create_conv23_project.tcl` 编译
- conv23 smoke M9K 分项（Fitter 实测）：**conv2 weight ROM 8 + conv3 weight ROM 9
  + 两个 bias ROM 共享 2 + FM RAM 4 = 23 块（50%）**；无 latch、无截断、无除法器/
  模运算器；该 9-bit 元素 9 为 retiming 前 conv23 smoke 结果，A+ 后完整 AC620
  Fitter 实测为 9-bit 元素 19 / DSP blocks 11（见本文件 A+ 小节）
- **修复（2026-08-02）**：未选权重 ROM 地址按层门控（`wt2_read = issuing &&
  !layer`、`wt3_read = issuing && layer`，conv3 运行期 conv2 ROM 地址原可达 4751
  超深 4608，现固定 0）；multiplier hex 注释勘误为 `32'h416335B9`/`32'h4D6CC8EC`
  （十进制值本就正确）；`tb_conv2_pool2`/`tb_conv3_serial` 新增 ROM 越界/门控断言

### 流式 GAP 流程（gap_stream_u8，2026-08-02 落地）

- 独立可综合流式 GAP：`rtl/gap_stream_u8.v`，直接消费 conv3_q 流（32×7×7、
  channel→y→x、每通道连续 49 项），每通道产出 1 个结果；**复用已冻结的
  `gap_div49.v`**（`q = floor((sum+24)/49)`，14 位无符号累加，max 12495）；冻结
  设计见 `docs/rtl_microarchitecture.md` §12
- 流式契约与 MaxPool 一致：输入计数/累加仅在 `busy && in_valid` 推进、支持任意
  空拍；每 49 项一个 `out_valid`；`out_channel` 严格 0..31、共 32 项；**与 conv3
  同启**（不由 conv3_done 启动）；busy 覆盖 S_DONE、done 单周期
- Questa 验证：`tb/tb_gap_stream_u8.v` 三遍（连续 32/32、无 reset 背靠背逐位
  相同、固定空拍 32/32），已并入 `run_questa.ps1`

### FC + Argmax 流程（fc_argmax_serial，2026-08-02 落地）

- 序列化 FC + signed argmax：`rtl/fc_argmax_serial.v`；`gap_mem[0:31]`（32×UINT8
  寄存器阵列，`gap_we` 写入）、`fc_weight` ROM 320×SINT8（OI）、`fc_bias` ROM
  10×SINT32；冻结设计见 `docs/rtl_microarchitecture.md` §13
- 数值：`fc_acc[o] = Σ gap_q[f]·fc_weight[o][f] + fc_bias[o]`，S64 精确累加、bias
  后 INT32 饱和、**不做 requant**；**tie 规则（冻结）**：class0 无条件初始化，
  后续仅 `acc32 > best` 更新（绝不用 `>=`），并列选更小类别号
- `prediction` done 时有效并保持到下次 start；支持无 reset 连续运行；每类
  PROLOGUE+32 MAC+ADD_BIAS+FC_VALID=35，10 类 + S_DONE = **351** 周期
- Questa 验证：`tb/tb_fc_argmax_serial.v` 三遍（golden 10/10 + prediction=8、
  无 reset 背靠背、**人工 tie** gap[0]=82 使类别 5/7 并列 6891 → 选 5），并重算
  signed argmax 比对；已并入 `run_questa.ps1`

### 完整纯计算核心流程（baseline_cnn_core，2026-08-02 落地）

- 完整端到端核心：`rtl/baseline_cnn_core.v`，实现 `input_q → stem → pool1 →
  conv2 → pool2 → conv3 → GAP → FC → Argmax → prediction`；控制器状态机
  `IDLE→STEM_START→STEM_WAIT→CONV2_START→CONV2_WAIT→CONV3_START→CONV3_WAIT
  →FC_START→FC_WAIT→DONE`；冻结设计见 `docs/rtl_microarchitecture.md` §14
- **启动规则（冻结）**：conv2+pool2 同一周期同启（layer_sel=0）、conv3+GAP 同一
  周期同启（layer_sel=1）、FC 在 32 个 gap 值写入后启动；conv2_done==pool2_done、
  conv3_done==gap_done 同周期；conv 引擎 `fm_raddr` 按控制器阶段选读 pool1/pool2
  RAM（每 RAM 非自身阶段读地址钳 0、数据 mux 阶段稳定）；conv2 q 直入 pool2、
  conv3 q 仅在 CONV3 阶段进 GAP
- **存储限制**：仅 input_q/pool1_q/pool2_q RAM + gap_mem 寄存器；**不实例化**完整
  stem_q/conv2_q/conv3_q RAM
- 核心级 busy/done：start 仅 IDLE 接受；busy 覆盖全部非 IDLE 状态；done 单周期；
  prediction done 时有效保持到下次 start；一次推理后无需 reset 可再次装载输入并
  start
- Questa 验证：`tb/tb_baseline_cnn_core.v` —— digit8 完整黄金 trace 11 节点全部
  逐位一致（12544/3136/6272/1568/1568/32/10/1，prediction=8）+ **10 smoke 样本
  digit0..digit9 无 reset 连续运行**（prediction 0..9 全对、每帧 fc_acc 逐位一致、
  每帧 done 恰一次）；A+ P=3 后总周期 **1,590,315**（旧值 1,529,163 +
  20,384×3，50 MHz 下 31.8063 ms），实测一致；已并入 `run_questa.ps1`
- Quartus smoke 工程：`baseline_cnn_core_smoke`（`quartus/baseline_cnn_core_smoke.
  {qpf,qsf}` 提交）；由 `scripts/create_full_core_project.tcl` 创建，
  `run_quartus_smoke.ps1 -ProjectName baseline_cnn_core_smoke -CreateScript
  create_full_core_project.tcl` 编译
- 完整核心 smoke M9K 分项（Fitter 实测）：**input RAM 1 + stem weight ROM 1 +
  pool1 RAM 4 + conv2 weight ROM 8 + conv3 weight ROM 9 + conv2 bias 2 + conv3
  bias 2 + pool2 RAM 1 + fc weight ROM 1 = 29 块（63%）**；fc bias ROM 落入逻辑
  （10 LC，同 stem bias）；LE 3,114、寄存器 928、9-bit 乘法器 19、无 latch、无
  截断、无实际 signed 警告、无除法器/模运算器
- `run_all.ps1` 现为 **14 步**：工具链 → 器件 → Questa（requant/GAP/stem/padding/
  maxpool/集成/conv2-pool2/conv3/GAP流/FC/完整核心）→ 算术 smoke → stem smoke →
  Python 契约 → stem 契约 → maxpool smoke → maxpool 契约 → 集成 smoke → conv23
  smoke → conv23 契约 → 完整核心 smoke → 完整核心契约

### A+ Requant 三级流水化（2026-08-02，50 MHz STA 通过）

- 新增 `rtl/requantize_u8_pipe.v`：固定 SAT32→S32×S32→S64 MUL→
  ROUND_SHIFT_SAT 三级；原 `requantize_u8.v` 不变并保留为 oracle
- stem/conv 在 ADD_BIAS 离开沿显式锁存 `acc64 + sign_extend(bias)` 及地址/last；
  EMIT、pipe out_valid、acc/q valid/data/address 严格同周期
- 周期：stem 188161（first q 14）、conv2 940801（first q 149）、conv3 460993
  （first q 293）、完整核心 1590315；三组 pool/GAP co-done 保持
- 局部 Questa：流水 requant、conv2+pool2、conv3、conv3+GAP、stem、padding、
  stem+pool1、完整核心 digit8+10 smoke、AC620 selftest（prediction=8）均通过
- `run_all.ps1` 14/14、完整 pytest 219 passed（1 条既有 PyTorch FutureWarning）
  均通过；未新增 false path 或 multicycle，SDC 仍为 20.000 ns
- 唯一一次 AC620 完整编译 Flow Successful：LE 2,933、寄存器 1,243、M9K 30、
  memory bits 166,784、9-bit 乘法器 19、DSP blocks 11、PLL 0、物理/虚拟引脚 5/0
- STA setup/hold：Slow 85C +2.274/+0.429 ns、Slow 0C +3.853/+0.400 ns、Fast 0C
  +12.380/+0.150 ns；最小 Fmax 56.41 MHz。retiming 前 `acc64[42] → GAP out_q[6]`
  的 -16.968 ns 长路径已消失；新最差 setup 为 conv3 权重 ROM 地址寄存器到
  `conv_u8_serial.acc64[63]`
- 设计已满足 50 MHz 时序烧录资格，但当前 `.sof` 来自未提交的 dirty worktree，
  仅作本地验证产物；提交前审查和用户再次确认前不得烧录，也不宣称已在真板运行

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
