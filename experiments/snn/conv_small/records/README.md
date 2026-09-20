# conv_small 实验记录索引

这里保留 `conv_small` 的原始实验记录。结构化输出和 checkpoint 仍位于
[`../results/`](../results/)；本次整理只移动 Markdown，不改历史数值，也不移动训练代码或结果目录。

| 记录 | 研究问题与主要变量 | 最重要结论 |
|---|---|---|
| [三 seed 冻结配置重跑](three_seed_record.md) | 原始 T=24 SNN，seed 7/17/27 | test 90.43% ± 0.20 pct；matched CNN 为 97.37%，差距不能简单归因于参数量。 |
| [线性时间量化](time_quantization_record.md) | T=24/12/8/5 | T=8/5 大幅降低 MAC 且准确率损失很小；更短 T 同时减少隐藏 IF 的跨时间积分。 |
| [T=4/T=3 扩展](T4_T3_time_quantization_record.md) | 线性量化 T=4/3 | T=4 仍接近 T=5，T=3 开始出现更明显损失；线性 T=4 输入严重前聚集。 |
| [最终膜电位读出](readout_experiment_record.md) | accumulated membrane 与完全等权 final membrane | 完全取消 early-fire advantage 会降低准确率并增加事件，但该实验也改变 logits scale，不能证明原偏置最优。 |
| [输入编码联合实验](input_encoding_joint_experiment.md) | T=4/5/8、Linear/Quantile、约 20.84%/30.18% firing ratio | Quantile 与约 30% firing ratio 有协同；T=4 组合 validation 最高、MAC 最低。 |
| [温和时间权重](temporal_weighting_record.md) | beta=1/0.5/0.25，固定总权重 | beta=0.5 validation 最高，温和削弱 early bias 可行。 |
| [事件正则化](event_regularization_record.md) | lambda=0/0.01/0.03/0.10 | 三个 λ 均满足精度预算；λ=0.10 的有效突触加法最低，较 λ=0 降 59.32%。 |
| [Frozen SNN 三 seed 正式结果](final_three_seed_record.md) | 最终 T=4 配置，seed 7/17/27；matched CNN 对比 | SNN test 93.47% ± 0.14 pct；CNN 为 97.37% ± 0.11 pct；事件代理均值 9,921.36/张。 |
| [器件曲线迁移](device_curve_transfer_record.md) | 三种单调器件变体，仅重校准 encoder | 9 次推理的最大 mean drop 为 0.52 pct；模拟范围内支持复用冻结权重。 |

跨实验、可长期复用的结论集中在 [`../experiment_history.md`](../experiment_history.md)。
