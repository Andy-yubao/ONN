# 8×8 光电导时间编码 Conv-SNN

## 目的

在上一轮 `64→128→10` 全连接 IF-SNN（测试准确率 87.15%）基础上，仅升级网络架构，
验证小型卷积 SNN 能否在相同 8×8 器件 latency coding 输入上达到 95% 测试准确率。

输入编码保持不变：

```text
MNIST 28×28 → 双线性缩放至 8×8 → 64 个光敏器件
→ 每个像素首次越过电导阈值时产生至多一个 spike → T=24
```

器件参数保持为 `G0=0.10`、`alpha=0.90`、`tau=5.0`、`G_threshold=0.35`。

## 网络与运行

```text
1×8×8
→ Conv 1→32, 3×3, padding=1 → LIF
→ Conv 32→64, 3×3, stride=2, padding=1 → LIF
→ Flatten 1024 → Linear 128 → LIF → Linear 10
→ 时间累计无泄漏膜电位 readout
```

神经元使用 `snntorch==1.0.0`、固定 `beta=0.9` 的极简 leak、reset-to-zero 和
fast-sigmoid surrogate gradient。`beta` 不学习，不增加模型参数；这是纯 IF 版本收敛平台后进行的
最后一次小调整。

在仓库根目录使用项目规定的 `onn` 环境运行：

```powershell
D:\tools\anaconda3\envs\onn\python.exe experiments/snn_8x8_conv_snn/train.py
```

默认训练配置为最多 10 epochs、batch size 256、Adam、学习率 0.001、随机种子 7；
测试准确率达到 95% 后自动停止。

## 本次结果

最终版本在 NVIDIA CUDA GPU 上完成 10 epochs 实测：

| 指标 | 结果 |
|---|---:|
| Final Test Accuracy | 91.27% |
| Best Test Accuracy | 91.27% |
| 95% 目标 | 未达到 |
| Total Training Time（最终版本） | 308.53 s |
| Average Time / Epoch | 30.85 s |
| Full Test Inference Time（10,000 张） | 2.772 s |
| Average Inference Time / Sample | 0.277 ms |
| Total Parameters | 151,072 |
| Model File Size | 610,165 bytes |
| Dense-equivalent MAC / time step | 445,696 |
| Dense-equivalent MAC / 24 steps | 10,696,704 |

训练损失从 1.1347 降至 0.2248，测试准确率从 83.90% 提升至 91.27%。结果正常收敛，
但未达到 95% 验收线。

本轮在主架构后共进行了 3 次人工小调整：去掉第二次 pooling、将剩余 pooling 改为
`stride=2` 卷积、最后将纯 IF 改为固定 `beta=0.9` 的极简 leak。按照任务停止条件，
不再继续搜索。

最可能的瓶颈是 8×8、每像素至多一次 spike 的稀疏输入经过多层离散 spike 传递时损失信息，
而不是模型参数不足。若后续只允许尝试一项修改，最值得加入可在推理时折叠进卷积/线性层的
归一化或阈值尺度稳定化，再检查能否跨过 95%。

结构化指标见 `results/metrics.json`，逐轮记录见 `results/history.csv`，训练曲线见
`results/training_curve.png`，最优权重见 `results/best_model.pt`。
