# 温和时间权重实验记录

状态：两次新增正式训练均已完成。固定 T=4、Quantile、约 30% input firing、seed=17；复用训练集
校准的 `g_threshold=0.21026152308606838` 和 quantile boundaries `[0,1,3]`，不重训 beta=1 baseline。

归一化公式为 `u_t=1+beta*(T-t)`、`w_t=u_t*10/sum(u)`。模型 logits 仍除以 T，
因此 beta=1 与历史 accumulated-membrane 实现数学等价，同时所有 beta 的权重和均为 10。

| beta | normalized temporal weights | Best Val (epoch) | Test | L1 fire | L2 fire | Synaptic adds |
|---:|---|---:|---:|---:|---:|---:|
| 1.0 | 4.0000 : 3.0000 : 2.0000 : 1.0000 | 95.44% | 93.88% | 0.2429 | 0.7028 | 23,758.61 |
| 0.5 | 3.5714 : 2.8571 : 2.1429 : 1.4286 | **95.66% (14)** | 94.35% | 0.2303 | 0.7801 | 23,204.61 |
| 0.25 | 3.1818 : 2.7273 : 2.2727 : 1.8182 | 95.58% (14) | 94.07% | 0.2295 | 0.7941 | 23,135.06 |

选择只看 validation：beta=0.5 最高，为 95.66%，因此进入事件正则化阶段。相对 beta=1，
validation 提高 0.22 pct，test 提高 0.47 pct，有效突触加法下降 2.33%。这支持在保留
early-fire advantage 时温和削弱其强度；beta=0.25 未进一步提高 validation。test 只在训练结束、
按 validation 选定 checkpoint 后评价一次，没有用于 beta 选择。

两次训练均为 20 epochs、Adam、batch=256、lr 1e-3→3e-4、weight decay=0、gradient clip=1；
各自 history 20 行且 final test 计数为 1，均正常退出。结构化结果见
[`beta_0p5_seed17/metrics.json`](../results/temporal_weighting/beta_0p5_seed17/metrics.json)和
[`beta_0p25_seed17/metrics.json`](../results/temporal_weighting/beta_0p25_seed17/metrics.json)。
