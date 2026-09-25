# 原始 28×28 MNIST 的 SNN/CNN 纯软件对照

## 固定协议

数据使用原始 MNIST 28×28 `ToTensor()`，顺序划分官方训练集的前 55,000 张为训练、后 5,000 张为验证；官方测试集 10,000 张只在每次训练根据验证集选出 checkpoint 后评估一次。两模型使用相同输入图像、划分、seed、20 epochs、batch 256、Adam、梯度裁剪 1.0、前 10 轮学习率 0.001、后 10 轮 0.0003。每轮只查看验证集，最佳验证准确率平局时取最早轮次。

两模型均为 `Conv(1→16, 3×3, stride 2) → Conv(16→32, 3×3, stride 2) → Linear(32×7×7→10)`，三个权重层都不使用 bias，参数量均为 **20,432**。CNN 在卷积后使用 ReLU，接收原始强度；SNN 在卷积后使用阈值为 1、subtract reset 的隐藏脉冲神经元，第一层和第二层均逐时间步更新，读出层累积加权脉冲电流。两者结构和参数规模匹配，输入表征与学习规则因模型类别而不同。

SNN 取历史实验中有利的 T=4、训练集校准的器件首次脉冲时间编码、quantile 映射、读出**时间权重** `beta=0.5`、事件正则系数 `lambda=0.10`。主对照的隐藏膜电位使用 IF（`decay=1.0`），与既有 8×8 冻结模型的神经元定义一致。事件正则归一化取 8×8 参考值 `20,876.389236363637 × (49 / 16)`，其中 49/16 是第二层空间位置数之比；这是预设的尺度近似，不是重新估计的 28×28 基线。事件计数表示有效连接上的脉冲加法次数，与 CNN 的稠密 MAC 不能视作相同硬件成本。

上述 T4/quantile/约 30% 输入编码、读出 beta 和事件正则的选择依据分别见[输入编码记录](../../snn/conv_small/records/input_encoding_joint_experiment.md)、[读出记录](../../snn/conv_small/records/temporal_weighting_record.md)和[事件正则记录](../../snn/conv_small/records/event_regularization_record.md)。28×28 的 18% 阈值是下面所述的原始像素约束下重新校准的结果。

## 28×28 编码的训练集约束

原始 28×28 图像的训练像素中，非零比例仅 **19.1202%**，所以 8×8 缩放图曾采用的约 30% 首次脉冲发放率在此无法达到。本实验仅用 55,000 张训练图的像素直方图确定 18% 目标，实际训练集发放率为 **17.9691%**，`g_threshold=0.15775603237773356`。基于训练集原始 24 步 latency 直方图，T4 的 quantile 边界为 `[0, 1, 2]`。第 0 步已有大量相同 latency 事件，不能拆分成等量时间 bin，因此 8×8 上的编码协同结论不能直接外推到原图。验证和测试图像不会重新校准。

## 运行

从仓库根目录使用 `onn` 环境运行：

```powershell
& 'D:\tools\anaconda3\envs\onn\python.exe' -m experiments.comparison.software_28.train --model snn --size 28 --decay 1.0 --seed 17 --epochs 20 --batch-size 256 --results-dir experiments/comparison/software_28/results/if_seed17 --final-test
& 'D:\tools\anaconda3\envs\onn\python.exe' -m experiments.comparison.software_28.train --model cnn --size 28 --seed 17 --epochs 20 --batch-size 256 --results-dir experiments/comparison/software_28/results/cnn_seed17 --final-test
```

上述两条命令分别将 `--seed` 和目录末尾的 `seed17` 同步替换为 `seed7`、`seed27` 各运行一次。最初的 LIF 探索分支对 seed 7、17、27 运行相同 SNN 命令，但使用 `--decay 0.5` 和 `results/snn_seed<seed>`。全部运行均在仓库根目录执行，未中途更改其他选项。

结果目录记录逐轮历史、配置、最佳 checkpoint 的哈希、验证/最终测试准确率以及 SNN 脉冲和有效加法统计。checkpoint 文件使用项目现有的 `*.pt` Git ignore 规则。

## 结果

三组预定 seed 均完成 20 轮训练。表中验证准确率为每个 seed 的最优 checkpoint，测试准确率为重载此 checkpoint 后的一次官方测试评估，单位均为 %。

| Seed | CNN 验证 | CNN 测试 | IF 验证 | IF 测试 | LIF `decay=0.5` 验证（探索） | LIF 测试（探索） |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 98.66 | 98.48 | 98.62 | 97.96 | 98.14 | 97.46 |
| 17 | 98.82 | 98.46 | 98.12 | 97.76 | 97.98 | 97.51 |
| 27 | 98.76 | 98.57 | 98.50 | 98.25 | 98.08 | 97.50 |
| 三 seed 均值 | **98.75** | **98.50** | 98.41 | 97.99 | 98.07 | 97.49 |

三 seed 中，CNN 对 IF 的平均测试准确率高约 **0.51 个百分点**。IF 的逐 seed 测试结果为 97.96%、97.76%、98.25%，不能由三次试验推断统计显著性。`decay=0.5` LIF 的测试均值为 97.49%，仅作误配分支记录。

| 模型 | Seed | 稠密等效 MAC/张 | 有效突触加法/张 |
|---|---:|---:|---:|
| CNN | 7 | 269,696 | 不适用 |
| CNN | 17 | 269,696 | 不适用 |
| CNN | 27 | 269,696 | 不适用 |
| CNN | 三 seed 均值 | 269,696 | 不适用 |
| IF SNN | 7 | 1,078,784 | 18,780.56 |
| IF SNN | 17 | 1,078,784 | 19,757.49 |
| IF SNN | 27 | 1,078,784 | 19,068.81 |
| IF SNN | 三 seed 均值 | 1,078,784 | 19,202.29 |
| LIF SNN（探索） | 7 | 1,078,784 | 16,990.33 |
| LIF SNN（探索） | 17 | 1,078,784 | 16,868.35 |
| LIF SNN（探索） | 27 | 1,078,784 | 15,470.45 |
| LIF SNN（探索） | 三 seed 均值 | 1,078,784 | 16,443.04 |

稠密等效 MAC 按模型全部权重连接计算；SNN 累加 4 个时间步。有效突触加法/张=`test_workload_total.effective_synaptic_additions / 10000`，由官方测试集的实际脉冲和有效连接扇出统计得到；CNN 不存在此项脉冲统计。这两列不能视作相同物理成本，也不据此作速度或能耗判断。逐组原始数据和均值见[结构化汇总](results/summary.json)。

执行顺序需保留：最初误将历史读出时间权重 `beta=0.5` 理解为隐藏膜泄露 `decay=0.5`，先训练并查看了该 LIF 分支与 CNN 的 28×28 测试集结果。发现语义混淆后，才确定以既有 8×8 IF 定义 `decay=1.0` 作为主对照。所有 `decay=0.5` 输出保留为探索性结果；后来 IF 分支虽然仍按验证集选择 checkpoint，但其测试结果发生在已接触同一测试集之后，应视为事后复核/探索性证据，不能称为严格独立 held-out 验证。IF 的选择依据是旧定义和参数语义，不是 28×28 测试分数。除这项纠正外，没有据测试分数更改预处理、模型规模、编码或训练预算。
