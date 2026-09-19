# SNN 线性 TTFS 时间量化实验记录

状态：T=12、8、5 三个配置均完成。日期：2026-09-18。仅 seed=17 的轻量探索，
不替代正式三 seed 复现，不选最终 champion。执行范围来自
当时的 `ONN_SNN_time_quantization_experiment_prompt.md`。

## 实现与冻结配置

- 编码器先按原器件模型在 24 步窗口内计算零基首次发放时间 old_t 和 -1 no-spike。
  保留其发放判定，再用 `q=floor(old_t*T/24)` 映射到 `[0,T-1]`；-1 不变。
  原本较晚发放的像素仍发一次，时间顺序单调，可以合并时间等级；不是截断窗口。
- 接口：`DeviceLatencyEncoder(time_steps=24, quantized_time_steps=T)`；
  原 time_steps 仍控制物理窗口，encoder.time_steps 是输出/网络展开长度。
  不传 quantized_time_steps 时保持旧行为；显式 T=24 也与原逻辑一致。
- 网络保持 Conv1→16 / IF → Conv16→32 stride2 / IF → Linear512→10，无 bias、9872 参数。
  IF threshold=1、严格大于阈值、subtract reset、reset spike detach、surrogate fast-sigmoid slope=5 不变。
- 器件 G0=0.10、alpha=0.90、tau=5、G_threshold=0.35 不变。
  时间加权膜电位读出递推不变，仍取 A[T-1]/T；压缩后使用目标 T。
- 复用 [统一协议](../../../common/README.md)：PIL bilinear 8×8 → ToTensor；
  顺序 train 55k / validation 5k，官方 test 10k。仅 seed=17。
- 三个新配置均 20 epochs、batch=256、num_workers=0、Adam、weight_decay=0、gradient_clip=1。
  epochs 1–10 学习率 1e-3，11–20 为 3e-4，CrossEntropyLoss；不加 spike penalty。
- 每轮只评价 validation，最大准确率选 checkpoint，平局选最早轮次；
  训练后加载它，validation 预热后仅一次完整 test，不根据 test 修改或重跑。
- onn 环境：Python 3.12.13、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、CUDA 12.8、RTX 5080 Laptop GPU；
  cuDNN deterministic=True、benchmark=False。当前 main 基线 commit
  `36b7467585183d476796b2747d098395a13143c1`，工作树含本次修改。
- T=24 全部指标直接取 [现有 seed17 metrics](../results/seed_17/metrics.json)，未重新训练或测试。

## 完整单张图片分类结果

| T | Seed | Best Val Acc | Test Acc | MAC/image | L1 fire/IF/image | L2 fire/IF/image | Synaptic additions/image |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 17 | 92.28% | 90.46% | 2,113,536 | 0.4289 | 5.2746 | 59,821.88 |
| 12 | 17 | 92.76% | 91.59% | 1,056,768 | 0.3326 | 2.5970 | 39,192.50 |
| 8 | 17 | 93.38% | 91.41% | 704,512 | 0.2828 | 1.6583 | 30,794.64 |
| 5 | 17 | 92.16% | 91.18% | 440,320 | 0.2197 | 1.0229 | 23,095.25 |

MAC 由模型函数实际生成/核对：单步均为 **88,064 MAC/图片/步**，完整图片乘目标 T。
该计数保留 padding，是当前稠密展开的理论加权累加数；当前 PyTorch 实现未事件跳零。
Synaptic additions 是完整 T 步的有效突触事件加法代理，排除 padding；不能替代 MAC 数。
两者均不含器件编码、IF/读出状态更新、访存和控制开销。

IF 发放均值按总发放数 / (10000×IF 数) 计算；L1 有 1024、L2 有 512 个 IF。
Neuron-step firing rate 再除 T；读出无阈值发放，不纳入隐藏 IF。

| T | Best epoch | Input events/image | L1 events/image | L2 events/image | L1 neuron-step rate | L2 neuron-step rate |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 18 | 13.3405 | 439.2343 | 2700.5811 | 1.79% | 21.98% |
| 12 | 19 | 13.3405 | 340.5518 | 1329.6618 | 2.77% | 21.64% |
| 8 | 19 | 13.3405 | 289.6109 | 849.0257 | 3.54% | 20.73% |
| 5 | 14 | 13.3405 | 224.9221 | 523.7171 | 4.39% | 20.46% |

四个 T 的输入发放总数都为 133405，即 **13.3405 events/image**，与压缩不丢失/不增加 spike 的契约一致。
隐藏层总 fire 次数下降不等于每个时间步更稀疏：第一层 neuron-step rate 上升，
第二层每步 rate 只小幅下降；较少时间步是总活动下降的重要原因。

## 输入 64 像素的逐时间步发放分布

对官方测试集全部 10000 张图片，仅执行与实验一致的预处理及 CUDA 器件编码，
未加载分类模型、未重新评价准确率；统计与训练 seed 无关。
表中“第 1 步”对应代码 q=0，“第 T 步”对应 q=T-1，均为压缩后的时间等级。
每步均值 = 该步全部图片中发放像素总数 / 10000。

| 压缩后的时间步 | T=8 平均 fire 像素/张 | T=5 平均 fire 像素/张 |
|---|---:|---:|
| 第 1 步 | 3.2746 | 8.2439 |
| 第 2 步 | 6.3530 | 3.8732 |
| 第 3 步 | 2.1199 | 0.8566 |
| 第 4 步 | 0.8475 | 0.2467 |
| 第 5 步 | 0.3787 | 0.1201 |
| 第 6 步 | 0.2467 | — |
| 第 7 步 | 0.1201 | — |
| 第 8 步 | 0.0000 | — |
| **发放像素合计** | **13.3405** | **13.3405** |
| **不发放像素（-1）** | **50.6595** | **50.6595** |
| **全部像素** | **64.0000** | **64.0000** |

T=5 的第 6–8 步不存在，故标记为“—”。每个像素至多 fire 一次；
两种编码均保留相同的 133405 次输入发放，变化的是时间分布。
分母包括每张图片的全部 64 像素，不是只对已发放像素归一化，也不是隐藏 IF 的 spike。
精确总数及均值保存在 [input_spike_timing.json](../results/time_quantization/input_spike_timing.json) 和
[CSV](../results/time_quantization/input_spike_timing.csv)。已核对每张平均发放/不发放合计为 64，
发放总数与各自既有最终测试记录一致。

## 运行时间（当前 GPU 环境）

| T | 20 轮训练秒数 | 完整测试秒数 |
|---:|---:|---:|
| 24 | 413.70 | 2.591 |
| 12 | 264.85 | 1.803 |
| 8 | 246.56 | 1.572 |
| 5 | 139.27 | 0.933 |

完整测试包含预处理、器件编码、前向、事件统计和 CUDA 同步，预热使用 validation。
GPU 负载及运行条件可能波动，这些记录不能作为 FPGA 性能或能耗证据。
T=5 时间只计完成的运行，不含中止的前缀训练。

## 简要分析与下一阶段候选

1. test 从 90.46% → 91.59% → 91.41% → 91.18%；时间压缩相对 T=24 本次没有测试准确率损失，
   三个新配置较基线分别高 1.13、0.95、0.72 个百分点。新配置内部压缩越强，test 略下降。
   validation 为 92.28% → 92.76% → 93.38% → 92.16%，并不单调。
2. T=12/8 均显著降低 MAC，同时 validation 不低于基线；T=5 的 validation 比基线低 0.12 个百分点，
   test 仍高于基线。单 seed 结果不证明统计显著性或跨 seed 稳健性。
3. 第二层平均 fire/IF/图片从 5.2746 降为 2.5970、1.6583、1.0229，随 T 减少而下降；
   T=5 平均仍大于 1，不能宣称所有 IF 已变为单次发放。未保存逐 IF 重复发放分布。
4. 有效突触加法从 59821.88 降为 39192.50、30794.64、23095.25 次/张；
   T=8 降 **48.52%**，T=5 降 **61.39%**。
5. **建议 T=8 进入下一阶段 spike-regularization 研究**：它有本轮最高 validation（93.38%），
   MAC 已降低 **66.67%**，第二层总 fire 降 **68.56%**。
   推荐依据验证集和活动/计算预算，不依据 test；T=5 可作为更低计算预算的参照。
   这里只提出后续候选，没有正式 champion，也没有自动开展下一阶段。

所有 SNN 配置 test 仍未达到 95%。本轮不改网络、loss、训练设置或新增 T/seed。
量化权重、RTL、FPGA 综合、板级和功耗：未运行。

## 输出、异常与验证

- [结构化汇总](../results/time_quantization/summary.json)保留全部公式口径、metrics 来源及 SHA256。
  新运行目录为 `results/time_quantization/T12_seed17/`、`T8_seed17/`、`T5_seed17_completed/`；
  每个保存 run_config.json、metrics.json、20 行 history.csv、training_curve.png 和本地 best_model.pt。
  源码/权重 SHA256 已核对；权重按 Git 规则忽略。
- 用户曾要求 T=8 后暂停，串行任务已自动启动 T=5，随后停止（观察到完成 4 轮、未做最终 test）。
  用户随后授权继续完成。原 checkpoint 仅有权重，无优化器/RNG 状态，因此在新目录同 seed 同配置重头完成。
  前四轮 loss/validation 与中止前一致。原 `T5_seed17/` 中的半成品与 interruption.json 保留，
  不参与结果表；这是用户中止后的恢复，不是依据 test 的重跑。
- 正式训练前 18 项检查通过：旧编码兼容、发放 mask/单次 spike/单调性、输出形状、
  T=12/8/5 前向、MAC 线性变化及现有共享协议/CNN/SNN 数值契约。
- 三个完整训练均正常退出；20 轮记录、验证集最优且平局取最早 checkpoint、
  运行前配置、源码和权重 SHA256、输入事件守恒核对通过。
- Git 状态读取有此前 pytest 临时缓存目录权限警告；未影响输出。本次 pytest 关闭缓存后通过。
- 修改文件为共享编码器、conv_small 训练入口与 README，新增时间量化契约测试及本记录/实验输出；
  网络 model.py 未修改。

## T=4 / T=3 后续扩展（2026-09-19）

用户随后明确要求新增 T=4、T=3；两者保持本记录的 accumulated-membrane 基线行为和 seed=17 协议。
完整训练、IF 平均发放、MAC/事件统计及输入像素逐时间步分布见
[T=4/T=3 扩展记录](T4_T3_time_quantization_record.md)。既有 T=8 下一阶段候选建议不变。
