# 实验记录模板

复制此模板为每组候选配置创建记录；运行前填写冻结配置，运行后填写实测结果。

## 运行前冻结

- 协议：`mnist8x8-sequential-v1`，见 [统一规则](README.md)
- 候选名称、架构、参数量：待填写
- seeds：`7, 17, 27`
- 数据：MNIST，顺序 55k/5k，官方 test 10k；PIL bilinear 8×8 → ToTensor
- 编码器及参数（CNN 填不适用）：待填写
- epochs、batch、优化器、学习率日程、weight decay、gradient clip：待填写
- 验证集架构/超参数选择过程和训练预算：待填写
- checkpoint：最高验证准确率，平局选最早 epoch
- 执行命令、Git commit、未提交修改说明：待填写
- 环境：onn；Python/PyTorch/torchvision/CUDA/GPU、确定性设置：待填写
- 输出目录：待填写（每 seed 独立，禁止覆盖）

## 运行后实测

| seed | 最优 epoch | validation % | final test % | 训练秒数 |
|---|---|---|---|---|
| 7 | 待填写 | 待填写 | 待填写 | 待填写 |
| 17 | 待填写 | 待填写 | 待填写 | 待填写 |
| 27 | 待填写 | 待填写 | 待填写 | 待填写 |

- validation/test 均值 ± 样本标准差（ddof=1）：待填写
- 每 seed 测试次数、是否发生测试后调参、exploratory 状态：待填写
- 参数量、checkpoint 文件大小、稠密等效操作、有效事件/突触操作：待填写
- 激活/状态存储及统计口径：待填写
- 计时口径：完整测试含预处理/编码、同步、统计；预热用 validation
- FPGA 资源预算/综合/板级/功耗：未运行（实际执行后更新）
- metrics/history/checkpoint 路径及失败/偏离规则记录：待填写
