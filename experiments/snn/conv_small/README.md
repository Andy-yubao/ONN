# 8×8 device-latency Conv-IF-SNN（small）

这是当前最重要的轻量 SNN 实验线，尚未选定 production champion。完整历史结论见
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

本轮单 seed 方向筛选选择：

- temporal beta=0.5，归一化权重 `3.5714:2.8571:2.1429:1.4286`；
- event lambda=0.10；
- validation 95.20%，final test 93.60%；
- 352,256 MAC/张，9,440.04 有效突触加法/张；
- L1/L2 为 0.0639/0.4388 fire/IF/张。

选择依据是相对 lambda=0 的 validation 损失 0.46 pct（不超过 0.5 pct）且事件成本最低。
这是 seed=17 的结构方向筛选结果，不是统计显著性验证，也不是 FPGA 能耗证据。

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
结束后 final test 只运行一次。当前正式结果位于 `results/temporal_weighting/` 与
`results/event_regularization/`，checkpoint 按仓库忽略规则保留在本地。
