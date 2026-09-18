# 8×8 MNIST 光电导时间编码 IF-SNN 最小实验

## 目的与数据流

本实验仅验证以下最小链路能否正常训练和分类：

```text
MNIST 28×28 → 双线性缩放至 8×8 → 光电导首次过阈值时间编码
→ 64→128→10 IF-SNN → 0~9 分类
```

## 器件与网络

每个像素模拟一个独立光敏单元，电导模型为：

```text
G(t) = G0 + alpha * p * (1 - exp(-t / tau))
```

离散窗口内首次达到 `G_threshold` 时产生一个 spike，未达到则不发放。实际参数为
`G0=0.10`、`alpha=0.90`、`tau=5.0`、`G_threshold=0.35`、`T=24`。

网络为 `64 → Linear → 128 IF（snntorch，surrogate gradient，reset-to-zero）→ Linear → 10`。
输出采用任务允许的无泄漏膜电位读出，避免额外引入 rate coding 或多种 readout 对比。

## 运行

在仓库根目录使用项目规定的 `onn` 环境：

```powershell
D:\tools\anaconda3\envs\onn\python.exe -m experiments.snn.mlp_latency_baseline.train
```

默认配置：5 epochs、batch size 512、Adam、学习率 0.002、随机种子 7。
原始数据默认写入仓库根目录的 `data/`，输出写入本实验目录下的 `results/`。训练时会在本地生成
`results/best_model.pt`；checkpoint 按仓库规则不纳入 Git 跟踪，提交的 JSON/CSV/PNG
是当前实验结果的持久记录。
本次环境安装并固定使用 `snntorch==1.0.0`。

## 探索性结果（不可作为 held-out benchmark）

在 NVIDIA CUDA GPU 上完成 5 epochs 实测。由于训练过程中逐 epoch 查看 test set，
并据此保存 best model，本结果仅作为 exploratory architecture baseline，不是严格
独立 held-out test benchmark：

| 指标 | 结果 |
|---|---:|
| Final Test Accuracy | 87.15% |
| Best Test Accuracy | 87.15% |
| Total Training Time | 65.69 s |
| Average Time / Epoch | 13.14 s |
| Full Test Inference Time（10,000 张） | 1.284 s |
| Average Inference Time / Sample | 0.128 ms |
| Total Parameters | 9,472 |
| Model File Size | 41,017 bytes |
| Dense weight accumulates / time step（上界） | 9,472 |

训练损失从 1.2521 降至 0.4024，测试准确率从 82.15% 提升到 87.15%，正常收敛。
sanity check 样本的亮度与首次 spike 时间相关系数为 -0.9528（仅统计会发放的像素），
且暗背景不发放，说明“越亮越早”的编码关系合理。

结构化指标见 `results/metrics.json`，逐轮记录见 `results/history.csv`，sanity check
见 `results/first_spike_sanity.png` 与 `results/sanity_check.json`。
