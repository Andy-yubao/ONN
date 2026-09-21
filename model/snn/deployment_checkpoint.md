# RTL deployment checkpoint

后续 RTL 默认且仅使用 seed 17 的 Frozen T=4 SNN checkpoint：

- 路径：`experiments/snn/conv_small/results/event_regularization/beta_0p5_lambda_0p10_seed17/best_model.pt`
- SHA256：`ecd4fa940c229bb9c717f33d68db2cdaf45ad6b83b48d597465d5431d8c845e6`
- best epoch：18；best validation accuracy：95.20%。

选择依据是三 seed 中最高的冻结 validation accuracy，以及原运行与确定性复跑得到完全相同
checkpoint SHA256。test accuracy 没有用于 checkpoint 选择。seed 7 和 seed 27 仍保留为稳健性
实验记录，但不是后续 RTL 的参数来源。

部署格式固定为 per-tensor symmetric INT8 weight、4 个隐藏状态 guard bits、INT17 readout
current，以及时间系数 `[5,4,3,2]`。可重复导出入口为 `python -m model.snn.export_params`，
正式产物与机器可读部署契约位于 [`export/`](export/)。导出器会先校验 checkpoint SHA256，
防止静默导出错误模型。

当前 checkpoint 和导出参数已被事件驱动稀疏 SNN RTL 使用，并通过 Python integer reference
到 RTL 的逐层与端到端仿真对拍。尚未创建最终 Vivado 工程，也未进行 synthesis、
implementation、资源/时序分析或 FPGA 板级验证。
