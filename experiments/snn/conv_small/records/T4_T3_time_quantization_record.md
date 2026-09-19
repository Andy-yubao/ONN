# T=4 / T=3 时间量化扩展实验记录

日期：2026-09-19。状态：T=4、T=3 均完成。两者只运行 seed=17、各 20 epochs，
是 [线性 TTFS 时间量化记录](time_quantization_record.md)的单 seed 探索扩展，不是正式多 seed 复现或 champion 选择。

## 可比性与运行协议

- 与既有 T=8/T=5 时间量化 baseline 使用相同 `accumulated_membrane` 读出：
  `R[t]=R[t-1]+I[t]`、`A[t]=A[t-1]+R[t]`、`logits=A[T-1]/T`。
- 当前工作树还包含之后的 final-membrane 实验。为避免混用，本轮新增显式 readout mode，
  训练命令指定 `--readout-mode accumulated_membrane`；运行前配置和结果均已核对。
- 其他设置保持不变：原 24 步器件发放判定，`q=floor(old_t*T/24)`，-1 保持不发放；
  相同 9872 参数网络、threshold/reset/surrogate、数据划分和预处理。
- seed=17；20 epochs；batch=256；Adam；weight decay=0；gradient clip=1；
  epochs 1–10 lr=1e-3，11–20 lr=3e-4。只按 validation 选最优 checkpoint，平局取最早；
  每个配置结束后仅做一次完整 test，无测试后调参。T=8/T=5 未重跑。
- T=4/T=3 的运行前冻结配置见 [T4_T3_extension_config.json](../results/time_quantization/T4_T3_extension_config.json)；
  精确训练源码快照及 SHA256 见 [manifest](../results/time_quantization/T4_T3_source_snapshot/manifest.json)。

## 准确率、计算量与平均 IF 发放

| T | Best val (epoch) | Test | MAC/张 | L1 fire/IF/张 | L2 fire/IF/张 | 突触加法/张 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 93.38% (19) | 91.41% | 704,512 | 0.2828 | 1.6583 | 30,794.64 |
| 5 | 92.16% (14) | 91.18% | 440,320 | 0.2197 | 1.0229 | 23,095.25 |
| 4 | 92.36% (20) | 91.07% | 352,256 | 0.1999 | 0.8442 | 20,738.68 |
| 3 | 91.62% (16) | 90.26% | 264,192 | 0.1688 | 0.6248 | 17,378.70 |

T=4 相对 T=5：MAC 再降 **20.00%**，
突触加法降 **10.20%**，
test 低 0.11 个百分点；validation 高 0.20 个百分点。T=3 相对 T=5：MAC 降
**40.00%**，突触加法降
**24.75%**，但 test 低 0.92、validation 低 0.54 个百分点。
这是单 seed 观察，不代表统计显著性。

每层 IF 均值 = 完整测试发放总数 / (10000 张 × 该层 IF 数)；L1 为 1024，L2 为 512。
每步发放率再除 T：

| T | 输入 events/张 | L1 events/张 | L2 events/张 | L1 neuron-step rate | L2 neuron-step rate |
|---:|---:|---:|---:|---:|---:|
| 8 | 13.3405 | 289.6109 | 849.0257 | 3.54% | 20.73% |
| 5 | 13.3405 | 224.9221 | 523.7171 | 4.39% | 20.46% |
| 4 | 13.3405 | 204.7364 | 432.2165 | 5.00% | 21.10% |
| 3 | 13.3405 | 172.8807 | 319.9151 | 5.63% | 20.83% |

随 T 降低，每张图片的总 fire 和突触加法继续下降；但 L1 每个 neuron-step 的发放率继续上升，
L2 每步发放率约保持在 20%–21%。因此总活动下降主要来自时间步减少，并不表示每个时间步全面更稀疏。
T=3 的 test 已低于 T=24 seed17 baseline 90.46%，开始显示过度压缩的准确率代价。
既有建议 **T=8 进入下一阶段 spike-regularization** 不变；若研究极限低计算预算，T=4 比 T=3 更合理。

## 64 个输入像素的逐时间步分布

只对官方 MNIST 测试集 10000 张图片运行相同预处理和器件编码；未加载模型，未重新评价准确率。
“第 1 步”对应编码 q=0。均值 = 该步输入发放总数 / 10000：

| 压缩时间步 | T=4 平均 fire 像素/张 | T=3 平均 fire 像素/张 |
|---|---:|---:|
| 第 1 步 | 9.6276 | 11.2458 |
| 第 2 步 | 2.9674 | 1.8508 |
| 第 3 步 | 0.6254 | 0.2439 |
| 第 4 步 | 0.1201 | — |
| **发放像素合计** | **13.3405** | **13.3405** |
| **不发放像素（-1）** | **50.6595** | **50.6595** |
| **全部像素** | **64.0000** | **64.0000** |

每个输入像素至多 fire 一次；T=4/T=3 的总输入发放均为 133405，和 T=24/12/8/5 一致。
这验证了量化仅改变时间分布，没有截断或新增输入 spike。精确总数和均值见
[input timing JSON](../results/time_quantization/T4_T3_input_spike_timing.json)及
[CSV](../results/time_quantization/T4_T3_input_spike_timing.csv)。

## 输出与验证

- T=4/T=3 输出分别位于 `results/time_quantization/T4_seed17/` 和 `T3_seed17/`，各含
  run_config.json、metrics.json、20 行 history.csv、training_curve.png 和本地 best_model.pt。
- 两次训练正常退出，测试各运行一次；checkpoint 选择、源码/checkpoint SHA256、MAC、有限数值、
  history 行数、读出模式和输入事件守恒均已核对。
- 训练前 25 项相关测试通过，包括 T=4/T=3 编码/前向/MAC 与两种读出模式契约。
- 训练入口当时把 network 名称和读出状态数写成 final-membrane 静态值，尽管实际 run_config 和模型为
  accumulated-membrane。训练后仅修正这两个元数据字段及派生状态更新数，未改变 checkpoint、准确率、
  spike/event 统计或其他实测结果；原始运行源码已快照。未来入口已改为按显式模式记录。
- 文档链接和 diff 格式检查通过。FPGA、量化、spike regularization：未运行。
