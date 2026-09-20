# Frozen T=4 SNN research model

冻结日期：2026-09-19。状态：验证阶段配置已冻结；后续验证不再调整网络、`T`、beta、
lambda 或 firing ratio。它是最终 SNN 研究候选，不等同于已经 promote 到 `model/` 的生产
champion。

## 模型与输入

```text
8×8 input
→ Conv 1→16, 3×3, padding=1, bias=False → subtract-reset IF
→ Conv 16→32, 3×3, stride=2, padding=1, bias=False → subtract-reset IF
→ Linear 512→10, bias=False
→ accumulated-membrane readout
```

- 参数量：**9,872**；IF threshold：`1.0`；训练 surrogate：fast sigmoid slope 5。
- 器件模型：`G(t) = G0 + alpha·p·(1-exp(-t/tau))`，`G0=0.10`、`alpha=0.90`、
  `tau=5.0`，24 步物理 crossing window。
- 输入编码：first-spike latency，`-1` 表示 no-spike；输出 `T=4`，Quantile mapping。
- firing threshold：`g_threshold=0.21026152308606838`；training-set firing ratio
  `30.0865625%`（19.2554 spikes/image）。
- Quantile boundaries：`[0, 1, 3]`，只由固定 55k training split 的 firing latency
  histogram 计算；validation/test 不参与校准。
- readout 时间权重：beta `0.5`，归一化权重
  `[3.5714288, 2.8571430, 2.1428571, 1.4285715]`。

## 训练协议

- seeds：`7, 17, 27`；每个 seed 20 epochs，batch size 256，Adam，weight decay 0；
- epoch 1–10：lr `1e-3`；epoch 11–20：lr `3e-4`；gradient clip norm `1.0`；
- event regularization：`lambda_event=0.10`；
- validation 最大值选择 checkpoint，同值取最早 epoch；final test 每个 seed 仅运行一次。

## 正式结果与产物

seed 17 的冻结结果为 Best Val **95.20%**、Test **93.60%**、L1/L2 activity
`0.0639/0.4388 fire/IF/image`、9,440.04 effective synaptic additions/image。确定性复跑
checkpoint SHA256 与原运行完全一致。

正式 checkpoint：

- seed 7：[`results/final_three_seed/seed_7/best_model.pt`](results/final_three_seed/seed_7/best_model.pt)；
- seed 17：[`results/event_regularization/beta_0p5_lambda_0p10_seed17/best_model.pt`](results/event_regularization/beta_0p5_lambda_0p10_seed17/best_model.pt)；
- seed 27：[`results/final_three_seed/seed_27/best_model.pt`](results/final_three_seed/seed_27/best_model.pt)。

完整三 seed 结果见 [final_three_seed_record.md](records/final_three_seed_record.md)。seed 7/27
训练时 Git commit 为 `16b549717d3f59650e7f1a8678295ae318250e26`；seed 17 原运行记录的 commit
为 `36b7467585183d476796b2747d098395a13143c1`。三者记录的冻结源码 SHA256 一致：

| Source | SHA256 |
|---|---|
| `experiments/snn/conv_small/train.py` | `6e2091598d1853d36bda0e01a073d2dd7496874b94c7211a01f7e77df90b79ed` |
| `experiments/snn/conv_small/model.py` | `5c4fa0508f2a8d9c8a1a350ed31e00db4e8a75aca03d6a822f0531f891d045c8` |
| `experiments/common/device_latency_encoder.py` | `cb4ca7026a5800ab177ac6b0b23450f19bb057532bb38ced3b7f9179de4207a9` |
| `experiments/common/comparison_protocol.py` | `a71e5a7438b3f814240847f8e3b55ec0c1ba06b063e80f6fef515a828300c0bfc` |

