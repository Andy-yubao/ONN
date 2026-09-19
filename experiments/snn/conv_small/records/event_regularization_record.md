# 隐藏层事件正则化实验记录

状态：三次新增正式训练均已完成。时间权重固定为阶段 A 选出的 beta=0.5；输入 baseline、
网络、IF dynamics 与训练协议全部冻结。

训练目标为 `CrossEntropy + lambda * R_event`。只惩罚 L1/L2 hidden spikes：

```text
raw proxy / image
= Σ(L1 spike × 该位置有效 Conv2 spatial fanout × 32 output channels)
 + Σ(L2 spike × 10 readout outputs)

R_event = raw proxy / image / 20,876.389236363637
```

固定 normalization constant 来自既有 beta=1 checkpoint 在 55k training split 上的一次只读统计；
未使用 validation/test 校准。forward 仍为真实 0/1 spike，regularizer 不 detach，梯度通过已有
surrogate 传播；固定输入 spike 不受惩罚。

| lambda | Best Val | Test | L1 fire | L2 fire | Hidden spikes/image | Synaptic adds | ΔVal | ΔSynaptic adds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 95.66% | 94.35% | 0.2303 | 0.7801 | 635.2723 | 23,204.61 | — | — |
| 0.01 | 95.38% | 93.76% | 0.1867 | 0.7417 | 570.9566 | 19,852.05 | -0.28 pct | -14.45% |
| 0.03 | 95.54% | 94.18% | 0.1253 | 0.6226 | 447.0956 | 14,837.94 | -0.12 pct | -36.06% |
| 0.10 | **95.20%** | 93.60% | **0.0639** | **0.4388** | **290.0512** | **9,440.04** | -0.46 pct | **-59.32%** |

以 lambda=0 的 validation 为基准，三个 lambda 的损失都不超过 0.5 pct；其中 lambda=0.10
有效突触加法最低，因此按预注册规则选中。test 不参与 lambda 选择。lambda=0.03 相对 0.01
同时有更高 validation 和更低事件成本；lambda=0.10 进一步降低事件，但 validation/test 较低，
形成明确的 accuracy–activity trade-off。

从 lambda=0 到 0.10，L1 fire/IF/张下降 72.27%，L2 下降 43.75%，说明 L1 更容易被压缩；
总 hidden spikes/张下降 54.34%。输入事件固定且未受惩罚。

各正则化运行在 validation 最佳 epoch 对应的训练集平均 loss 为：

| lambda | best epoch | CE | normalized R_event | total loss |
|---:|---:|---:|---:|---:|
| 0.01 | 12 | 0.1788 | 0.8073 | 0.1869 |
| 0.03 | 18 | 0.1747 | 0.5738 | 0.1919 |
| 0.10 | 18 | 0.1924 | 0.3190 | 0.2243 |

三次均为 20 epochs，每次 final test 计数为 1，均正常退出。结构化结果位于
[`results/event_regularization/`](../results/event_regularization/)。

## 用户追加的同配置确认运行

在上述五次方向筛选完成后，用户明确追加一次 `T=4 / Quantile+约30% / beta=0.5 /
lambda=0.10 / seed=17` 重跑。确认运行同样完成 20 epochs，best validation 95.20%
（epoch 18）、final test 93.60%、有效突触加法 9,440.0364/张。其 checkpoint SHA256
与首次运行完全相同；逐轮 CE、R_event、total loss、validation 和 learning rate 也完全相同
（仅耗时字段不同）。这验证了当前环境中的确定性复现，不是新增 seed，不能替代多 seed 稳健性验证。

确认结果见
[`beta_0p5_lambda_0p10_seed17_rerun/metrics.json`](../results/event_regularization/beta_0p5_lambda_0p10_seed17_rerun/metrics.json)。
