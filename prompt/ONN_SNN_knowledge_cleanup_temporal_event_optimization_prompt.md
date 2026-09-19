# ONN：SNN 实验知识整理 + 时间权重与事件稀疏化实验提示词

你现在需要在 ONN 项目中继续优化 `experiments/snn/conv_small`。本次任务分成两部分：

1. 先整理此前多轮实验积累下来的知识与实验记录；
2. 在当前最佳输入编码候选基础上，依次做“时间权重温和化”和“隐藏层事件稀疏化”。

严格控制复杂度：本次最多新增 **5 次正式训练**，全部固定 `seed=17`。

---

## 1. 开始前：先整理实验记录，不运行训练

先检查：

- `experiments/snn/conv_small/`
- `README.md`
- `three_seed_record.md`（若存在）
- `time_quantization_record.md`
- `readout_experiment_record.md`
- `input_encoding_joint_experiment.md`
- `results/` 中的重要结构化结果
- `docs/development_plan.md`

不要假设文件名完全一致，先以实际仓库为准。

整理目标：

> 保留所有原始实验记录，同时建立一个清晰的索引，以及一份“我们已经学到了什么”的长期文档。

不要删除历史实验数据，不修改历史数值，不做大规模目录重构。

若现有结构允许，建议整理成：

```text
experiments/snn/conv_small/
├─ README.md
├─ experiment_history.md
├─ records/
│  ├─ README.md
│  ├─ three_seed_record.md
│  ├─ time_quantization_record.md
│  ├─ readout_experiment_record.md
│  ├─ input_encoding_joint_experiment.md
│  └─ ...
├─ model.py
├─ train.py
└─ ...
```

若移动 Markdown，必须同步更新仓库内所有相对链接并通过 link check。不要为“整洁”去移动训练代码、结构化结果或 checkpoint。

`records/README.md` 用作实验索引：每份记录研究了什么、主要变量是什么、最重要结论是什么。

---

## 2. 新建 `experiment_history.md`，提炼此前真正获得的知识

这份文件不要复制全部原始表格，而要记录经过实验支持的长期结论。

至少写入以下内容。

### 2.1 原始轻量 SNN 基线

```text
8×8 input
→ Conv 1→16 → IF
→ Conv 16→32 stride2 → IF
→ Linear 512→10
```

参数约 `9,872`，与 matched CNN 基本相同。

原始 `T=24` 三 seed：

```text
seed 7  test ≈ 90.22%
seed 17 test ≈ 90.46%
seed 27 test ≈ 90.62%
mean    ≈ 90.43%
```

matched CNN test mean 约 `97.37%`。

结论：

> SNN 与 CNN 的差距不能简单归因于参数量不足。

### 2.2 时间量化实验

关键数据：

```text
T=8 test ≈ 91.41%, MAC=704,512
T=5 test ≈ 91.18%, MAC=440,320
T=4 test ≈ 91.07%, MAC=352,256
T=3 test ≈ 90.26%, MAC=264,192
```

结论：

- `T=24 → 8/5/4` 准确率损失很小，而 MAC 大幅下降；
- `T=3` 开始出现更明显损失；
- 降低 T 不只是降低 latency 分辨率，也减少隐藏 IF 的跨时间积分机会。

### 2.3 原线性时间量化的前聚集问题

例如 `T=4`：

```text
step1 ≈ 9.63 pixels/image
step2 ≈ 2.97
step3 ≈ 0.63
step4 ≈ 0.12
```

结论：

> 原线性量化下，大多数有效输入 spike 挤在最早时间步，时间维度利用率低。

### 2.4 输入 firing ratio 的经验

原输入 firing ratio 约 `20.84%`，后提高到约 `30.18%`。

结论：

> 单独增加 firing ratio 收益有限；其价值与时间分布方式强耦合。

### 2.5 Quantile temporal encoding 的关键经验

联合实验：

```text
T=4 + Quantile + ~30%
Val  = 95.44%
Test = 93.88%
MAC  = 352,256

T=5 + Quantile + ~30%
Val  = 94.96%
Test = 93.98%

T=8 + Quantile + ~30%
Val  = 95.26%
Test = 93.91%
```

结论：

1. Quantile 单独使用没有提高准确率；
2. 单独提高 firing ratio 也只有有限收益；
3. **Quantile + ~30% firing ratio 存在明显协同**；
4. `T=4` validation 最高且 MAC 最低，因此最适合作为下一阶段研究 baseline；
5. 三个组合 test 极接近，单 seed 下不能过度解释 0.1 pct 级差异。

### 2.6 完全等权 readout 的失败经验

曾将 accumulated-membrane readout 改为 final-membrane / equal temporal weight，结果：

```text
test: 91.41% → 89.84%
L2 firing: +51.84%
synaptic additions: +20.07%
```

结论：

> 完全取消 early-fire advantage 不好。

同时记录方法学限制：

> 那次实验同时改变了 temporal weight shape 与 logits scale，因此不能证明当前 early-time bias 已经最优，只能证明“直接完全等权”不合适。

### 2.7 当前架构理解

当前每个 SNN time step 内执行完整前馈：

```text
input spike(t)
→ Conv1
→ IF1 update
→ Conv2
→ IF2 update
→ Linear
→ output update
```

然后才进入 `t+1`。

因此：

- `T` 不是网络传播深度需要的时间；
- `T` 是神经元状态进行跨时间积分的次数；
- Conv 负责提取空间关系；
- IF membrane 负责整合不同时间到达的空间证据。

### 2.8 器件曲线与 Quantile 的开放假设

记录但不要在本次实验验证：

> 若真实器件的光强→latency 曲线发生单调变形，但仍保持“光强越大、越早 crossing”，则重新校准 firing threshold 与 quantile boundaries 后，可能无需重新训练整个 SNN。

---

## 3. 后续实验统一固定当前输入 baseline

固定：

```text
T = 4
mapping = Quantile
input firing ratio ≈ 30%
seed = 17
g_threshold = 0.21026152308606838
```

当前参考：

```text
Best validation accuracy = 95.44%
Final test accuracy      = 93.88%
MAC/image                = 352,256
L1 fire/IF/image         = 0.2429
L2 fire/IF/image         = 0.7028
Synaptic additions/image = 23,758.61
```

复用现有 training-set quantile boundaries。

不要重新校准输入编码，不要重新训练 baseline。

---

# 4. 阶段 A：时间权重温和化

## 4.1 目标

当前 `T=4` accumulated-membrane readout 等价时间权重为：

```text
4 : 3 : 2 : 1
```

已知完全改成 `1:1:1:1` 不好。

本阶段只验证：

> 在保留 early-fire advantage 的情况下，适度减弱其强度是否更好。

## 4.2 参数化

定义：

```text
u_t(beta) = 1 + beta * (T - t)
```

`T=4` 时：

```text
beta=1.00 → 4.00 : 3.00 : 2.00 : 1.00   # 当前 baseline
beta=0.50 → 2.50 : 2.00 : 1.50 : 1.00
beta=0.25 → 1.75 : 1.50 : 1.25 : 1.00
```

但不能直接使用 raw weights。

必须归一化，使每组：

```text
sum(weights) = 10
```

与 baseline `4+3+2+1=10` 一致：

```text
w_t(beta) = u_t(beta) * 10 / sum(u(beta))
```

要求：

- `beta=1.0` 与当前实现数学等价；
- 只改变 temporal weight shape；
- logits 总体尺度保持一致。

## 4.3 训练规模

现有 `beta=1.0` 直接作为 baseline，不重训。

只新增：

```text
beta=0.5
beta=0.25
```

共 **2 次训练**。

训练协议与当前正式配置完全一致：

```text
seed=17
epochs=20
batch size=256
Adam
lr=1e-3 → 3e-4
weight decay=0
gradient clip=1.0
```

每个实验按 validation 选择 checkpoint，训练结束后 final test 只做一次。

## 4.4 选择规则

以 validation 为主：

- 若 `beta=0.5/0.25` 优于或不劣于 `beta=1.0`，选 validation 最好的进入阶段 B；
- 若二者均更差，则继续使用 `beta=1.0`。

不要根据 test 选择 beta。

---

# 5. 阶段 B：隐藏层事件稀疏化

## 5.1 目标

输入编码与准确率已改善后，再研究：

> 是否可以在尽量保持分类能力的情况下，减少 L1/L2 hidden spikes 与有效突触加法。

## 5.2 不要重写 Adam

继续使用 Adam。

这里的“自定义优化”指：

```text
自定义训练目标 / event-aware regularization
```

而不是新写一个 Adam/RMSProp 类优化器。

## 5.3 Loss

从：

```text
L = CrossEntropy
```

改为：

```text
L = CrossEntropy + lambda * R_event
```

`R_event` 只惩罚：

```text
Layer1 hidden spikes
Layer2 hidden spikes
```

不要惩罚固定 input spikes。

## 5.4 Event cost proxy

优先实现一个与当前 `effective synaptic additions` 方向一致的**可微 proxy**。

原因：

- L1 一个 spike 会驱动 Conv2 的多个有效连接；
- L2 一个 spike 只驱动 Linear；
- 两层单个 spike 的事件成本不同。

可以复用已有 spatial fanout / connection-count 逻辑，让 L1/L2 spike 按后续有效连接数近似加权。

要求：

1. forward 仍使用真实 0/1 spikes；
2. regularizer 的梯度通过现有 surrogate gradient 传播；
3. regularizer 中不要对 spike `.detach()`；
4. 不建立复杂 FPGA 能耗模型；
5. 目标只是一个简单、可解释的 event-aware differentiable proxy。

## 5.5 归一化

不能把数量级数万的 raw event count 直接加到 CE 上。

将 `R_event` 固定归一化到 baseline 训练时约 `O(1)` 的数量级。

归一化必须：

- 简单；
- 固定；
- 可解释；
- 不使用 validation/test 做校准。

在实验记录中写清公式和 normalization constant。

## 5.6 lambda

只试：

```text
lambda = 0.01
lambda = 0.03
lambda = 0.10
```

共 **3 次训练**。

全部使用阶段 A 选出的固定 temporal weighting。

不要追加更多 lambda。

---

# 6. Event regularization 选择规则

不能只追求最少 spike。

以进入阶段 B 时的 `lambda=0` validation accuracy 为参考：

1. 将 validation accuracy 损失控制在：

```text
≤ 0.5 percentage point
```

2. 在满足此 accuracy budget 的方案中，优先选：

```text
effective synaptic additions/image 最低
```

的方案。

若三个 lambda 全部使 validation 下降超过 0.5 pct，则保留：

```text
lambda=0
```

不要根据 test 选择 lambda。

---

# 7. 本次训练总量

```text
阶段 A：2 次
阶段 B：3 次
总计：5 次
```

不要：

- 重跑已有 baseline；
- 增加 seed；
- 增加 beta；
- 增加 lambda；
- 做 final confirmation rerun。

本阶段是结构方向筛选，不是统计显著性验证。

---

# 8. 每次实验记录

至少记录：

### Accuracy

- best validation accuracy
- best epoch
- final test accuracy

### Activity

- L1 fire / IF / image
- L2 fire / IF / image
- total hidden spikes / image

### Workload

- MAC / image
- effective synaptic additions / image

### Event-regularized run 额外记录

- CE loss
- normalized `R_event`
- total loss
- lambda

不需要新增复杂图表。

---

# 9. 新增实验记录文件

整理后的记录区中新增：

```text
temporal_weighting_record.md
event_regularization_record.md
```

### temporal_weighting_record.md

至少：

| beta | normalized temporal weights | Best Val | Test | L1 fire | L2 fire | Synaptic adds |
|---:|---|---:|---:|---:|---:|---:|
| 1.0 | baseline | 95.44% | 93.88% | 0.2429 | 0.7028 | 23,758.61 |
| 0.5 | ... | ... | ... | ... | ... | ... |
| 0.25 | ... | ... | ... | ... | ... | ... |

### event_regularization_record.md

至少：

| lambda | Best Val | Test | L1 fire | L2 fire | Hidden spikes/image | Synaptic adds | ΔVal | ΔSynaptic adds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | selected temporal baseline | ... | ... | ... | ... | ... | — | — |
| 0.01 | ... | ... | ... | ... | ... | ... | ... | ... |
| 0.03 | ... | ... | ... | ... | ... | ... | ... | ... |
| 0.10 | ... | ... | ... | ... | ... | ... | ... | ... |

---

# 10. 实验完成后再次更新 `experiment_history.md`

只加入实验真正支持的新知识，例如：

- moderate early-time bias 是否优于当前 `4:3:2:1`；
- 完全等权失败后，温和削弱是否可行；
- event regularization 能否在 ≤0.5 pct validation loss 下显著降低 synaptic additions；
- L1/L2 哪一层更容易被压缩；
- 是否出现明确 accuracy–activity Pareto trade-off。

不要把假设写成事实。

---

# 11. README 与 development plan

## README

保持简洁，只保留：

- 当前网络；
- 当前研究 baseline；
- 当前最重要指标；
- 如何运行主实验；
- 指向 `experiment_history.md` 与 `records/README.md` 的链接。

不要继续把所有历史实验直接堆进 README。

## docs/development_plan.md

做必要更新：

- 输入编码联合实验：已完成；
- temporal weighting：本轮状态；
- event regularization：本轮状态；
- 器件迁移 / 反色 / FPGA 等仍未开始。

---

# 12. 本轮严格禁止

不要修改：

- `T=4`
- Quantile boundaries
- input firing ratio
- `g_threshold`
- 网络结构
- IF threshold
- neuron dynamics
- reset
- surrogate slope
- batch size
- learning rate
- epochs
- optimizer 类型
- 数据划分

不要新增：

- leak
- refractory period
- 多 seed
- 更多 beta
- 更多 lambda
- equal-weight readout
- inverted MNIST
- 器件 transfer 实验
- FPGA
- quantization
- 大规模目录重构

不要删除任何历史实验数据。

---

# 13. 轻量测试要求

正式训练前至少验证：

1. `beta=1.0` 的 generalized weighting 与当前 baseline 数学等价；
2. 不同 beta 的 normalized weight sum 相同；
3. forward shape 不变；
4. 参数量不变；
5. MAC 不变；
6. event regularizer 可反向传播；
7. `lambda=0` 与无 regularizer 行为等价；
8. 整理后的 Markdown 链接有效。

不要建立庞大测试体系。

---

# 14. 最终报告

完成后按以下顺序报告：

## A. 实验记录整理

- 整理了哪些文件；
- 新目录结构；
- `experiment_history.md` 中沉淀了哪些核心知识。

## B. 时间权重实验

- `beta=1 / 0.5 / 0.25` 对比；
- 最终选择哪个 beta；
- 是否支持“温和削弱 early-fire advantage”。

## C. 事件稀疏化实验

- `lambda=0 / 0.01 / 0.03 / 0.10` 对比；
- accuracy–activity trade-off；
- 是否找到满足 ≤0.5 pct validation loss 的更低事件成本方案。

## D. 当前结论

- 当前最合理的研究 baseline；
- 当前仍未解决的问题；
- 不要宣布 production champion。

## E. 验证

- 测试数量；
- 文档链接检查；
- 是否出现中止、异常或无效训练。

完成后停止，不要自动进入器件迁移、反色识别或 FPGA 阶段。
