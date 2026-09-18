# 8×8 器件时间编码轻量 IF-SNN

本实验实施 [`docs/snn_next_model_architecture.md`](../../docs/snn_next_model_architecture.md)
中的首选架构，保持现有器件模型、输入预处理和 `T=24` 不变。

## 数据流与网络

```text
MNIST 28×28
→ PIL 双线性 Resize 至 8×8 → ToTensor
→ G(t) 首次过阈值时间编码（每像素至多一个 spike）
→ Conv 3×3, 1→16, padding=1 → IF（threshold=1, subtract）
→ Conv 3×3, 16→32, stride=2, padding=1 → IF（threshold=1, subtract）
→ Flatten 512 → Linear 10
→ 无 leak 的 R/A 时间加权膜电位读出 → 分类
```

隐藏 IF 更新严格执行“同一步积分、严格大于阈值发放、减阈值复位”，复位分支的
spike 指示量 detach；fast-sigmoid slope=5 只用于训练反向传播。输出 logits 为
`A[23]/24`，推理 argmax 可省略这个固定正比例。

器件参数为 `G0=0.10`、`alpha=0.90`、`tau=5.0`、`G_threshold=0.35`、`T=24`。
官方 MNIST 训练集固定按顺序划分为 55,000 张训练集和 5,000 张验证集，测试集
10,000 张仅用于最终评估；默认训练 20 epochs、batch size 256、seed 7，前 10 轮
学习率 `1e-3`，后 10 轮为 `3e-4`。

## 运行

在仓库根目录使用项目规定的 `onn` 环境：

```powershell
D:\tools\anaconda3\envs\onn\python.exe experiments/snn_8x8_device_if_conv_small/train.py
```

输出写入 `results/`，包括验证集曲线、验证集最优 checkpoint、最终测试指标和有效
突触加法工作量代理。权重文件遵循仓库的忽略规则，不作为代码提交。

## 本次实测结果

在 `onn` 环境使用 seed=7 完成默认 20 epochs 训练；第 15 轮验证集最优，随后只用
该 checkpoint 评估测试集：

| 指标 | 结果 |
|---|---:|
| Best validation accuracy | 92.32%（epoch 15） |
| Final test accuracy | 90.22% |
| 95% 目标 | 未达到 |
| Total training time | 436.62 s |
| Full test inference time（含编码、CUDA 同步） | 2.702 s |
| Average inference time / sample | 0.270 ms |
| Total parameters | 9,872 |
| Dense-equivalent MAC / time step | 88,064 |
| Dense-equivalent MAC / 24 steps | 2,113,536 |
| Effective synaptic additions / test sample | 56,408.83 |

结果表明该首版达到 90% 低资源候选门槛，但尚未达到 95% 目标。量化、FPGA 综合、
RTL 仿真、板级测试和功耗测量均未运行，不能据此声称硬件资源或能耗结果。

结构化指标见 `results/metrics.json`，逐轮曲线数据见 `results/history.csv`，训练曲线
见 `results/training_curve.png`，验证集最优权重见 `results/best_model.pt`。
