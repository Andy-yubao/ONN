# 神经网络模型

## 目录结构

```
model/
├── README.md                       # 本文件
├── requirements.txt                # Python 依赖
├── configs/                        # 训练与模型配置文件
│   ├── baseline_cnn.json
│   └── tiny_resnet.json
├── onn_model/                      # 核心 Python 包
│   ├── __init__.py
│   ├── data.py                     # MNIST 数据管线与划分
│   ├── engine.py                   # 训练循环、检查点、评测
│   ├── metrics.py                  # 准确率、参数统计
│   ├── reproducibility.py          # 随机种子、环境捕获
│   ├── profiling.py                # 参数量与 MAC 估算
│   ├── activity.py                 # 通道活动分析
│   └── models/
│       ├── __init__.py
│       ├── baseline_cnn.py         # BaselineCNN 定义
│       └── tiny_resnet.py          # Tiny-ResNet 定义
├── train.py                        # 训练入口
├── evaluate.py                     # 评测入口
├── profile_model.py                # 模型规模与 MAC 分析
├── analyze_activations.py          # 通道活动分析
├── scripts/
│   └── run_m1_baselines.ps1        # 全流程脚本
└── tests/
    ├── test_models.py              # 模型形状、梯度测试
    ├── test_data.py                # 数据划分测试
    └── test_training_smoke.py      # 训练烟雾测试
```

## 环境与依赖

```bash
conda activate onn
```

或：

```bash
pip install -r model/requirements.txt
```

## 运行命令

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
# PowerShell 换行
python model/evaluate.py `
    --config model/configs/tiny_resnet.json `
    --checkpoint model/runs/TinyResNet/<run-id>/best.pt
```

### 模型规模与 MAC 分析

```bash
python model/profile_model.py --model BaselineCNN --markdown
python model/profile_model.py --model TinyResNet --markdown
```

### 通道活动分析

```bash
python model/analyze_activations.py `
    --config model/configs/tiny_resnet.json `
    --checkpoint model/runs/TinyResNet/<run-id>/best.pt `
    --output model/profiles/activity.json
```

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

## 常见错误

| 错误 | 排查 |
|------|------|
| `ModuleNotFoundError: No module named 'onn_model'` | 确保在项目根目录运行命令 |
| MNIST 下载失败 | 检查代理：`set HTTP_PROXY=http://127.0.0.1:10809` |
| `CUDA out of memory` | 减小 batch-size，或使用 `--device cpu` |
| 训练结果无法完全复现 | 确认 `config.json` 中 `"deterministic": true`。固定随机种子可提高可重复性，但非确定性 CUDA 算法仍可能产生小幅差异。严格对照实验应启用 deterministic 模式；日常快速实验可关闭确定性以提高性能。 |

## 工作流程

### B 轨：28×28 CNN 数字推理（基线已建立 ✅）

1. ✅ 在 `onn_model/models/` 中实现 BaselineCNN 和 Tiny-ResNet
2. ✅ 配置文件和超参数保存在 `configs/`
3. ✅ 评估浮点基线准确率
4. 📋 后续：实现定点/量化参考模型
5. 📋 后续：导出定点权重供 FPGA 使用

### C 轨：器件曲线优化探索（待启动）

1. （基线建立后）基于论文器件曲线实现自定义优化器
2. 与普通 Adam 基线对照比较
3. 用未来实测曲线替换论文参数

> 普通 Adam CNN 必须作为 FPGA 推理基线和器件曲线方法对照。不得预设器件曲线优化器一定优于 Adam。
