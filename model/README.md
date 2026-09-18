# `model/`：正式模型与部署准备

当前尚未选择 production / deployment champion。所有候选网络位于
[`experiments/`](../experiments/)，完成统一 8×8 CNN/SNN 公平比较并正式完成 champion
selection 之前，不将候选模型 promote 到本目录；若最终选择 SNN，再按本目录契约继续
SNN 的量化、整数参考和部署导出。

未来 `model/` 只承载正式模型及其部署契约：

- frozen architecture；
- trained checkpoint metadata；
- quantization；
- fixed-point / integer reference；
- hardware parameter export；
- deployment-oriented tests。

当前尚未开始 SNN 量化或正式部署导出。
