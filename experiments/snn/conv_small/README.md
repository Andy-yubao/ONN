# 8×8 device-latency Conv-IF-SNN（small）

这是当前最重要的轻量 SNN 实验线；验证阶段配置已经
[冻结](frozen_model.md)，但尚未 promote 为 `model/` 中的 production champion。完整历史结论见
[实验知识沉淀](experiment_history.md)，逐轮实验与原始记录见 [实验记录索引](records/README.md)。

## 网络与固定输入 baseline

```text
MNIST 28×28 → bilinear 8×8
→ frozen device first-spike latency encoding
→ Conv 1→16 → IF
→ Conv 16→32 stride2 → IF
→ Linear 512→10 temporal readout
```

网络无 bias，共 9,872 参数；IF threshold=1、strict `>`、subtract reset、无 leak，训练反向使用
fast-sigmoid surrogate slope=5。当前研究固定 `T=4`、Quantile、约 30% input firing ratio、
`g_threshold=0.21026152308606838` 和 training-set boundaries `[0,1,3]`。

冻结配置的三 seed 正式验证结果：

- temporal beta=0.5，归一化权重 `3.5714:2.8571:2.1429:1.4286`；
- event lambda=0.10；
- Best-Val mean 94.97%，test **93.47% ± 0.14 pct**（sample std）；
- 352,256 dense-equivalent MAC/张，平均 9,921.36 effective synaptic additions/张；
- 平均 L1/L2 activity 为 0.0691/0.4506 fire/IF/张。

三 seed 详情和 matched CNN 对比见 [正式记录](records/final_three_seed_record.md)。三种模拟
器件曲线的 encoder-only 迁移最大 mean accuracy drop 为 0.52 pct，见
[迁移记录](records/device_curve_transfer_record.md)。事件工作量仍不是 FPGA 能耗证据。

## 运行主实验

从仓库根目录使用 `onn` 环境：

```powershell
D:\tools\anaconda3\envs\onn\python.exe -m experiments.snn.conv_small.train `
  --input-preset t4_quantile_030 `
  --readout-mode accumulated_membrane `
  --temporal-beta 0.5 `
  --event-lambda 0.10 `
  --seed 17 --epochs 20 --batch-size 256 --learning-rate 0.001 `
  --num-workers 0 --device cuda --results-dir <new-empty-directory>
```

训练入口拒绝覆盖非空结果目录。每轮只评价 validation，以最大 validation 选 checkpoint；
结束后 final test 只运行一次。正式三 seed 结果位于 `results/final_three_seed/` 与 seed 17
的 `results/event_regularization/beta_0p5_lambda_0p10_seed17/`；迁移结果位于
`results/device_curve_transfer/`。checkpoint 按仓库忽略规则保留在本地。
