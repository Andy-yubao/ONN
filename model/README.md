# 神经网络模型

> 本文档说明模型代码、入口命令与当前状态。数值细节与硬件格式分别见
> [模型架构](../docs/model_architecture.md)、[量化数据格式](../fpga/baseline_cnn/docs/data_format.md)
> 与 [硬件目标](../fpga/baseline_cnn/docs/hardware_target.md)。

## 目录结构

```
model/
├── README.md                       # 本文件
├── requirements.txt                # Python 依赖（torch>=2.0、torchvision>=0.15、numpy>=1.24、matplotlib>=3.7、pytest>=7.0）
├── configs/                        # 训练与模型配置文件
│   ├── baseline_cnn.json
│   ├── tiny_resnet.json
│   ├── micro_cnn_s.json            # M2 compact 配置
│   ├── micro_cnn_xs.json
│   └── ds_micro_cnn.json
├── onn_model/                      # 核心 Python 包
│   ├── __init__.py
│   ├── data.py                     # MNIST 数据管线与划分
│   ├── engine.py                   # 训练循环、检查点、评测
│   ├── metrics.py                  # 准确率、参数统计
│   ├── reproducibility.py          # 随机种子、环境捕获
│   ├── profiling.py                # 参数量与 MAC 估算
│   ├── activity.py                 # 通道活动分析
│   ├── quantization.py             # 量化原语（含 BN 融合）
│   ├── int8_ptq.py                 # INT8 训练后量化（W8A8 仿真）
│   ├── int8_reference.py           # 纯整数参考模型（冻结数值标准）
│   └── models/
│       ├── __init__.py
│       ├── baseline_cnn.py         # BaselineCNN（FPGA 部署模型）
│       ├── tiny_resnet.py          # Tiny-ResNet（软件参考）
│       └── compact_cnn.py          # M2 compact 模型
├── train.py                        # 训练入口
├── evaluate.py                     # 评测入口
├── profile_model.py                # 模型规模与 MAC 分析
├── analyze_activations.py          # 通道活动分析
├── export_baseline_cnn_hardware.py # 硬件参数导出（FPGA 部署链路）
├── verify_bn_fusion.py             # BN 融合验证
├── verify_baseline_int8_reference.py # 纯整数参考模型验证
├── verify_baseline_cnn_hardware_export.py # 导出正确性验证
├── evaluate_baseline_int8_ptq.py   # INT8 PTQ 评估
├── scripts/
│   ├── run_m1_baselines.ps1        # M1 基线全流程
│   ├── run_m2_compact_models.ps1   # M2 compact 全流程
│   ├── _multiseed_training.py      # 多种子训练
│   ├── _collect_multiseed.py       # 多种子结果收集
│   ├── _evaluate_checkpoints.py    # 检查点批量评测
│   └── _profile_all.py             # 批量 profile
└── tests/                          # pytest 测试（含 RTL 契约测试）
    ├── test_models.py              # 模型形状、梯度测试
    ├── test_data.py                # 数据划分测试
    ├── test_training_smoke.py      # 训练烟雾测试
    ├── test_quantization.py        # 量化原语测试
    ├── test_bn_fusion.py           # BN 融合测试
    ├── test_int8_ptq.py            # INT8 PTQ 测试
    ├── test_int8_reference.py      # 整数参考模型测试
    ├── test_hardware_export.py     # 参数导出测试
    ├── test_stem_rtl_contract.py   # stem RTL 契约（地址/tap/常数）
    ├── test_maxpool_rtl_contract.py# MaxPool RTL 契约
    ├── test_conv23_rtl_contract.py # conv2/conv3 RTL 契约
    ├── test_full_core_rtl_contract.py # 完整核心 RTL 契约
    └── test_rtl_vector_contract.py # 黄金向量契约
```

## 环境与依赖

```bash
conda activate onn
```

或：

```bash
pip install -r model/requirements.txt
```

> 仓库内没有 venv，一律使用 `onn` conda 环境（`D:\tools\anaconda3\envs\onn`，torch 2.11.0+cu128）。
> Git Bash 直接调用：`/d/tools/anaconda3/envs/onn/python.exe <script>`

## 运行命令

所有命令从**仓库根目录**运行。

### 训练 BaselineCNN

```bash
# PowerShell
python model/train.py --config model/configs/baseline_cnn.json

# 覆盖参数
python model/train.py --config model/configs/baseline_cnn.json --epochs 2 --batch-size 64 --device cpu
```

### 训练 Tiny-ResNet

```bash
python model/train.py --config model/configs/tiny_resnet.json
```

### 烟雾测试（快速验证管线）

```bash
python model/train.py --config model/configs/baseline_cnn.json --smoke-test
python model/train.py --config model/configs/tiny_resnet.json --smoke-test
```

### 评估最佳检查点

```bash
python model/evaluate.py --config model/configs/tiny_resnet.json --checkpoint model/runs/TinyResNet/<run-id>/best.pt
```

### 模型规模与 MAC 分析

```bash
python model/profile_model.py --model BaselineCNN --markdown
python model/profile_model.py --model TinyResNet --markdown
```

### 通道活动分析

```bash
python model/analyze_activations.py --config model/configs/tiny_resnet.json --checkpoint model/runs/TinyResNet/<run-id>/best.pt --output model/profiles/activity.json
```

### 测试

```bash
python -m pytest model/tests
```

## FPGA 部署链路（BaselineCNN，已完成）

> 冻结数值标准：`Int8Reference` + `candidate_quant_config.json`（方案 A，per-tensor 权重）。
> 数据格式见 [data_format.md](../fpga/baseline_cnn/docs/data_format.md)。

| 步骤 | 命令 / 入口 | 状态 |
|------|------------|:---:|
| 浮点基线训练 | `model/train.py` | ✅ |
| BN 融合验证 | `python model/verify_bn_fusion.py` | ✅ |
| INT8 PTQ 评估 | `python model/evaluate_baseline_int8_ptq.py` | ✅ |
| 整数参考验证 | `python model/verify_baseline_int8_reference.py` | ✅ |
| 硬件参数导出 | `python model/export_baseline_cnn_hardware.py` | ✅ |
| 导出验证 | `python model/verify_baseline_cnn_hardware_export.py` | ✅ |
| 完整 pytest | `python -m pytest model/tests`（含 RTL 契约测试） | ✅ |

导出产物位于 `fpga/baseline_cnn/params/`（`.mem`/`.mif`/`.vh` + `checksums.sha256`）；RTL、仿真与板级验证见 `fpga/baseline_cnn/docs/rtl_microarchitecture.md` 与 `hardware_target.md`。

## 输出目录

每次运行输出到 `model/runs/<model-name>/<timestamp>/`：

```
config.json          # 配置快照
environment.json     # 运行环境信息
history.csv          # 逐 epoch 训练/验证指标
metrics.json         # 最佳验证指标与参数量
best.pt              # 最佳验证检查点
last.pt              # 最后一个 epoch 检查点
```

> `model/runs/` 与 `model/data/` 已 gitignore；权重仅本地。

## 常见错误

| 错误 | 排查 |
|------|------|
| `ModuleNotFoundError: No module named 'onn_model'` | 确保在项目根目录运行命令 |
| MNIST 下载失败 | 检查代理：`set HTTP_PROXY=http://127.0.0.1:10809` |
| `CUDA out of memory` | 减小 batch-size，或使用 `--device cpu` |
| 训练结果无法完全复现 | 确认 `config.json` 中 `"deterministic": true`。固定随机种子可提高可重复性，但非确定性 CUDA 算法仍可能产生小幅差异。 |

## 工作流程

### B 轨：MNIST 数字推理（已完成）

1. ✅ 在 `onn_model/models/` 中实现 BaselineCNN 和 Tiny-ResNet
2. ✅ 配置文件和超参数保存在 `configs/`
3. ✅ 评估浮点基线准确率（含多种子审计）
4. ✅ BN 融合、INT8 PTQ、纯整数参考模型
5. ✅ 导出定点权重/偏置/测试向量供 FPGA 使用
6. ✅ RTL bit-accurate 比对、50 MHz STA、固定 digit8 板级自检

### 新增研究方向（2026-08-02 组会，待设计）

- 器件模型驱动训练：标准 RGB 图像 → 电导/电流表示 → 训练与评价
- 定向 Hebbian 连接增强（公式 TBD）
- KAN 候选（结构 TBD）
- 多次采样与时序特征编码（编码 TBD）

> 以上方向处于决策/待设计阶段，参数为开放问题（见 [open_questions.md](../docs/open_questions.md) NQ 清单），不预设结论。
