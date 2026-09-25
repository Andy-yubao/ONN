# 实验记录

`experiments/` 只承载候选方案、失败实验、消融实验和 CNN/SNN 公平比较，不承载正式
production 模型或 FPGA 工程。

```text
experiments/
├── common/                         # 共享器件编码与轻量契约测试
├── snn/
│   ├── mlp_latency_baseline/       # 原 snn_8x8_device_encoding
│   ├── conv_large/                # 原 snn_8x8_conv_snn
│   ├── conv_small/                # 原 snn_8x8_device_if_conv_small, 最重要的基线
│   └── lif_if/                    # 冻结 8×8 SNN 的 IF/LIF 消融
├── cnn/                            # matched 8×8 CNN baseline
└── comparison/                     # 软件公平比较与 28×28 实验
```

各 SNN 实验保留 README、源码和对应的实验结果记录。
其中 `snn/conv_small/`（原 `snn_8x8_device_if_conv_small`）是当前最重要的 SNN 基线，
后续 matched CNN 公平比较和 champion selection 优先以它为参考；若公平比较最终选择
SNN，才将选定的 SNN promote 到 `model/`；
本次只重构路径与共享依赖，不重新生成或修改历史结果数值。训练时会在本地生成
`results/best_model.pt`，checkpoint 按仓库规则不纳入 Git 跟踪；JSON/CSV/PNG 是提交的
结果持久记录。

MLP latency baseline 的 87.15% 和 large Conv-SNN 的 91.27% 都是在训练过程中逐轮查看
test set 并据此选择 checkpoint 的 exploratory 结果，不能作为严格独立 held-out test
benchmark。`conv_small` 使用验证集选择 checkpoint，测试集只用于最终评估。

推荐从仓库根目录使用 `onn` 环境运行：

```powershell
D:\tools\anaconda3\envs\onn\python.exe -m experiments.snn.mlp_latency_baseline.train
D:\tools\anaconda3\envs\onn\python.exe -m experiments.snn.conv_large.train
D:\tools\anaconda3\envs\onn\python.exe -m experiments.snn.conv_small.train
```

正式 champion 只有在统一 8×8 CNN/SNN 比较后才进入 [`model/`](../model/)。
