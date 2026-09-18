# ONN 项目 AI 开发代理指令

> 本文件是项目规则的唯一权威来源。`CLAUDE.md` 只路由到本文件，不维护独立副本。

## 会话约定

- 所有回复使用中文；
- 不得虚构未运行的验证；
- 以当前 checkout 的活跃分支和工作树为基线，不擅自切换到旧分支重新开始；
- 不覆盖或删除未提交的用户修改。

## 当前项目主线

ONN 当前围绕以下 promotion pipeline：

```text
8×8 输入 / 器件动态模型
→ first-spike latency encoding
→ matched CNN/SNN 公平实验
→ champion selection（若选择 SNN，再 promote）
→ model/ 中的量化、整数参考、硬件导出
→ fpga/ 中的 RTL / 仿真 / 综合 / 板级验证
```

当前研究目标是在统一 8×8 输入、统一数据划分和相近参数规模下公平比较 CNN 与 SNN；
不得在 matched CNN 完成前声称 SNN 已优于 CNN、已证明最佳或已满足精度要求。

## 四个核心目录

| 目录 | 职责 |
|---|---|
| `experiments/` | 候选方案、失败/消融实验、公平比较 |
| `model/` | 正式 champion、量化、整数参考、硬件导出 |
| `fpga/` | 正式模型的 FPGA 实现 |
| `history/` | 历史阶段索引和 Git tag 入口，不复制旧源码 |

当前三个 SNN 实验位于 `experiments/snn/`。其中 `experiments/snn/conv_small/` 是当前
最重要的 SNN 基线，原实验目录名为 `snn_8x8_device_if_conv_small`；后续 matched CNN
公平比较、模型选择和部署讨论优先以它为参考。共同编码器位于
`experiments/common/device_latency_encoder.py`。编码参数和 `-1` no-spike 约定以实现和
实验记录为准，不复制 encoder fallback。

## 当前硬件边界

- AC620 / Cyclone IV 是 legacy platform；
- Basys3 / Artix-7 是当前优先候选目标；
- champion、位宽、状态存储和资源预算冻结前，不创建最终 Vivado 工程或新的 RTL；
- 不把旧 AC620 固定样本自检扩展为当前 SNN 的硬件部署证据。

## 开发与验证约定

- 必须使用 `onn` Conda 环境：`D:\tools\anaconda3\envs\onn`；
- 从仓库根目录运行 Python module；
- 是否训练、评估、量化或运行硬件流程以当前任务范围和阶段要求为准；不得把未运行的流程写成已完成；
- matched CNN/SNN 研究中，test set 只用于最终评价；逐轮查看 test set 的历史结果必须标为 exploratory；
- 修改文档后运行 `python tools/check_markdown_links.py`；
- 运行过的命令和测试才可以写成验证结果，未运行内容必须明确标记为“未运行”。

## 历史锚点

旧 28×28 CNN、INT8、纯整数参考和 AC620 FPGA 工程由
`pre-snn-era-restructure-20260918` tag 保存。不要复制到 `history/`，不要删除该 tag，
也不要对包含该历史的分支 force push。
