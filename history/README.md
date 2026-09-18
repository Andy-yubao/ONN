# 历史阶段索引

当前 active tree 已切换到 8×8 器件动态模型、first-spike 编码、CNN/SNN 公平比较和
后续 SNN-to-FPGA 主线。旧时代源码没有复制到 `history/`；完整 Git 历史由锚点保存：

```text
git tag pre-snn-era-restructure-20260918
```

## Legacy 28×28 CNN / AC620 阶段

旧阶段曾完成 28×28 MNIST CNN 软件基线、BaselineCNN / Tiny-ResNet、BN fusion、INT8
PTQ、纯整数参考、硬件参数导出、完整 CNN RTL、模块级/集成级仿真、50 MHz STA，以及
AC620 / Cyclone IV 固定样本板级自检。冻结整数 BaselineCNN 的原生 28×28 测试精度约为
98.47%；本索引不重新运行这些实验。

## 28×28 模型与低有效分辨率过渡实验

冻结的 28×28 BaselineCNN 无法直接兼容低有效分辨率输入：原生 28×28 约 98.47%，
8×8 → 28×28 约 31.37%，4×4 → 28×28 约 10.45%。这些数字是旧实验记录的摘要，
不是当前主线的模型结论；完整结果与评估代码请通过上述 tag 查看。

完整历史源码、文档和 FPGA 工程请查看 `pre-snn-era-restructure-20260918`。当前 active
tree 不再维护旧 28×28 CNN / AC620 实现，但旧成果并未删除，它被 Git 历史永久保存。
