# 实验记录

`experiments/` 只承载候选方案、失败实验、消融实验和 CNN/SNN 公平比较，不承载正式
production 模型或 FPGA 工程。

```text
experiments/
├── common/                         # 共享器件编码与轻量契约测试
├── snn/
│   ├── mlp_latency_baseline/       # 原 snn_8x8_device_encoding
│   ├── conv_large/                # 原 snn_8x8_conv_snn
│   └── conv_small/                # 原 snn_8x8_device_if_conv_small
├── cnn/                            # matched 8×8 CNN baseline 规划
└── comparison/                     # 公平比较协议
```

三个 SNN 实验保留原有 README、源码、`metrics.json`、`history.csv`、PNG 和小型测试；
本次只重构路径与共享依赖，不重新生成或修改历史结果数值。训练时会在本地生成
`results/best_model.pt`，checkpoint 按仓库规则不纳入 Git 跟踪；JSON/CSV/PNG 是提交的
结果持久记录。

推荐从仓库根目录使用 `onn` 环境运行：

```powershell
D:\tools\anaconda3\envs\onn\python.exe -m experiments.snn.mlp_latency_baseline.train
D:\tools\anaconda3\envs\onn\python.exe -m experiments.snn.conv_large.train
D:\tools\anaconda3\envs\onn\python.exe -m experiments.snn.conv_small.train
```

正式 champion 只有在统一 8×8 CNN/SNN 比较后才进入 [`model/`](../model/)。
