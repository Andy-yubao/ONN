# CNN/SNN 统一比较协议

协议 ID：`mnist8x8-sequential-v1`。实现见 [comparison_protocol.py](comparison_protocol.py)。

## 固定规则

- 正式重复实验只用三个 seed：`7, 17, 27`。CNN/SNN 按相同 seed 配对；不挑选最好 seed。
- seed 控制 Python、NumPy、PyTorch、CUDA 初始化和训练 shuffle；cuDNN deterministic=True、benchmark=False。
  相同软件/硬件环境下复现；不承诺跨版本、跨设备逐位一致。
- 官方 MNIST 训练集按原始顺序：索引 `[0,55000)` 训练、`[55000,60000)` 验证；
  官方测试集全部 10000 张。三个 seed 不改变划分。
- 在 PIL 图像上双线性 Resize 至 8×8，再 ToTensor，像素为 `[0,1]`；不加归一化或数据增强。
  SNN 在此基础上使用共享 [器件编码器](device_latency_encoder.py)，CNN 直接使用图像。
- 训练集 shuffle=True；验证/测试 shuffle=False。当前基线 batch=256、num_workers=0。
- checkpoint 只按最高 validation accuracy 选择；相同准确率保留最早 epoch。
- 架构/超参数选择只用验证集。配置冻结后，每个 seed 的最佳 checkpoint 仅做一次完整测试；
  计时预热使用验证集。不得根据测试结果调整配置后将重测称为独立最终评价。
- 历史上逐轮查看 test set 的结果必须标为 exploratory；原 small SNN seed=7 单次结果保留为历史记录，
  不混入本次三 seed 汇总。
- 汇总全部三个 seed 的准确率均值和样本标准差（ddof=1），同时保留各 seed 数值。
  三次重复仅描述波动，不据此声称统计显著优势。
- CNN 参数目标约 8k–12k；参考 SNN 9872。训练算法可因模型而不同，但须公开预算、优化器、
  学习率、checkpoint 规则及验证集选择过程，不能将相同 seed 解读为相同权重或完全相同 shuffle。

## 输出与记录

每次运行写入独立目录，如 `results/seed_7/`，禁止覆盖非空目录。保存：
`metrics.json`、`history.csv`、`training_curve.png`、本地 `best_model.pt`（权重不纳入 Git）。
记录按 [实验记录模板](experiment_record_template.md) 填写。

后续比较至少报告准确率、参数量、模型存储、稠密等效操作、有效事件/突触操作、
激活/状态存储与预计 FPGA 资源压力。GPU 时间仅为当前运行环境的参考；硬件未运行项写“未运行”。
