# ONN 项目

光学神经网络（ONN）FPGA 部署研究项目。GitHub：https://github.com/Andy-yubao/ONN

模型代码位于 `model/`（独立可交付模块）；`experiments/` 存放实验记录与结果。

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

## 模型与分支基线

- 模型定义在 `model/onn_model/models/`：
  - **M1 交付候选**：`BaselineCNN`（第一 FPGA 部署候选）、`TinyResNet`（高准确率备选 / 软件参考）
  - **M2 compact**：`MicroCNNSmall` / `MicroCNNExtraSmall` / `DepthwiseMicroCNN`（轻量化探索）
- 当前模型开发基线分支：`model/m1-baseline-audit`；**不要从 `master` 重新开始**
- 历史权重（`model/runs/` 本地 .pt）已验证可用当前模型定义以 strict=True 加载
