# SNN 最终膜电位读出单变量实验

日期：2026-09-18。状态：完成。范围来自
当时的 `ONN_SNN_final_membrane_readout_experiment_prompt.md`。
仅训练一个 T=8、seed=17 模型，未重训或重新测试 baseline，未追加配置。

## 唯一行为修改与冻结协议

原读出为 `R[t]=R[t-1]+I[t]`、`A[t]=A[t-1]+R[t]`、`logits=A[T-1]/T`。
T=8 时电流权重为 `(8,7,6,5,4,3,2,1)/8`。
本次保留 R 的积分，删除 A，直接使用 `logits=R[T-1]=sum_t I[t]`，权重全部为 1。
这是提示指定的原始求和读出；除时间权重变化外，logits 的尺度也随之变化，
本次未另做尺度归一化，不能把训练后的差异完全归因于时间偏置一个机制。

网络仍为无 bias 的 Conv1→16/IF、Conv16→32 stride2/IF、Linear512→10，9,872 参数。
IF threshold=1、strict threshold、subtract reset、reset spike detach、fast-sigmoid slope5 不变。
器件参数、24 步物理窗口、`floor(old_t*8/24)` 时间映射、-1 no-spike 不变。
编码器和共享比较协议源码 SHA256 与 baseline 相同。

沿用固定 PIL bilinear Resize(8,8)/ToTensor、顺序 train 55,000 / validation 5,000、test 10,000。
20 epochs、batch=256、num_workers=0、Adam、weight_decay=0、CrossEntropyLoss、gradient clip=1；
前 10 轮 lr=1e-3，后 10 轮 3e-4。不修改预处理、loss、结构或其他超参数。
每轮只看 validation，最大验证准确率选 checkpoint，平局选最早；
结束后加载 best，validation 预热，仅一次完整 test，同时收集活动统计，无 test 调参或重跑。

运行于 onn 环境、CUDA、RTX 5080 Laptop GPU；Python 3.12.13、PyTorch 2.11.0+cu128、
torchvision 0.26.0+cu128，cuDNN deterministic=True、benchmark=False。
当前 main commit 为 `36b7467585183d476796b2747d098395a13143c1`，工作树有既有及本次修改。

## 对比结果

baseline 引用 [现有 T8_seed17 metrics](../results/time_quantization/T8_seed17/metrics.json)
及 [时间量化记录](time_quantization_record.md)，新结果见
[final membrane metrics](../results/readout_experiment/T8_seed17_final_membrane/metrics.json)。

| Readout | T | Seed | Best Val Acc | Test Acc | MAC/image | L1 fire/IF/image | L2 fire/IF/image | Synaptic additions/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| accumulated membrane baseline | 8 | 17 | 93.38% | 91.41% | 704,512 | 0.2828 | 1.6583 | 30,794.64 |
| final membrane readout | 8 | 17 | 91.92% | 89.84% | 704,512 | 0.3066 | 2.5179 | 36,976.23 |

新方案最佳轮次为 epoch 14（baseline 为 19），20 轮训练耗时 406.05 s，最终测试 2.709 s。
输入发放总数仍为 133,405（13.3405 events/image）。L1 为 313.9332 events/image，
L2 为 1289.1413 events/image；neuron-step firing rate 分别为 3.8322%、31.4732%。

MAC 仍为 88,064/步 × 8；有效突触加法沿用有效连接事件统计，排除 padding，
不是 MAC、FPGA 资源或功耗。隐藏 IF 发放按总发放/(test 张数×IF 数)计算。
读出状态从 R/A 两组 20 个降为 R 一组 10 个，状态更新从 160 降为 80/张；
此项不计入上述 MAC 与有效突触加法。

## 四个问题的结论

1. accuracy 未提高：validation 降 1.46 个百分点，test 降 1.57 个百分点。
2. spike activity 增加：L1 相对增加 8.40%，L2 相对增加 51.84%，第二层变化明显。
3. 有效突触加法增加 6,181.59/张，即 20.07%。
4. 不建议作为后续 T=8 baseline：验证准确率下降且隐藏活动/事件计算预算变差；
   这一建议基于 validation 和活动预算，test 仅作为最终报告。
   保留本次实现和输出作为实验记录；历史 accumulated-membrane 结果仍为推荐参照。

单 seed 探索不证明统计显著性、跨 seed 稳健性或 CNN/SNN 胜负；未达到 95% 目标。
不开展额外 seed/T、尺度对照、quantile binning、spike regularization、量化或 FPGA 工作。

## 输出与验证

运行命令（仓库根目录）：

```powershell
D:\tools\anaconda3\envs\onn\python.exe -m experiments.snn.conv_small.train --time-steps 8 --seed 17 --epochs 20 --batch-size 256 --learning-rate 0.001 --device cuda --results-dir experiments/snn/conv_small/results/readout_experiment/T8_seed17_final_membrane
```

独立输出包含 run_config.json、metrics.json、20 行 history.csv、training_curve.png 和本地 best_model.pt。
训练正常退出，test_evaluations=1；20 轮、验证选择、权重哈希、源码哈希、输入事件守恒核对通过。
checkpoint 按现有忽略规则不纳入 Git；历史输出和既有用户修改保留。

正式训练前运行 conv_small 模型、共享编码器、时间量化、比较协议、CNN 模型相关测试，
`pytest -q -p no:cacheprovider`：19 passed。
新增 T=8 非零读出检查，捕获每步电流并核对 logits 等于其等权和，且不同于旧读出；
shape=[B,10]、参数数与 MAC 契约通过。
文档链接检查 `python tools/check_markdown_links.py` 通过，无失效相对链接。

训练无数值或执行异常。Git 状态采集仍有此前 pytest 临时缓存目录权限警告，未影响训练和结果。
本轮训练耗时高于旧记录，未调查运行负载原因，不能归因于读出改动或作为硬件性能结论。
权重量化、RTL、综合、板级及功耗均未运行。
