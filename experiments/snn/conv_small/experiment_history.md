# conv_small 实验知识沉淀

本文件只提炼已有实验支持的长期结论；完整表格、运行异常和结构化输出见
[实验记录索引](records/README.md)。单 seed 的 0.1 个百分点级差异不作过度解释。

## 原始轻量 SNN 与 matched CNN

```text
8×8 input
→ Conv 1→16 → IF
→ Conv 16→32 stride2 → IF
→ Linear 512→10
```

模型约 9,872 参数，与 matched CNN 基本相同。原始 `T=24` 三 seed test 分别约为
90.22%（seed 7）、90.46%（seed 17）、90.62%（seed 27），均值约 90.43%；
matched CNN test 均值约 97.37%。因此当前 SNN–CNN 差距不能简单归因于参数量不足，
也尚未据此选定 production champion。

## 时间量化与输入时间分布

线性量化的 seed=17 结果为：T=8 test 91.41%、704,512 MAC；T=5 test 91.18%、
440,320 MAC；T=4 test 91.07%、352,256 MAC；T=3 test 90.26%、264,192 MAC。
从 T=24 压缩到 T=8/5/4 的准确率损失很小而 MAC 大幅下降，T=3 开始出现更明显损失。
降低 T 不只降低 latency 分辨率，也减少隐藏 IF 跨时间积分的机会。

线性 T=4 的输入 spike 每张图约为 9.63、2.97、0.63、0.12，绝大多数集中在最早时间步，
说明时间维度利用率低。原输入 firing ratio 约 20.84%，提高后约 30.18%；单独增加 firing
ratio 收益有限，其价值与时间分布方式强耦合。

## Quantile temporal encoding

固定约 30% firing ratio 的联合实验得到：T=4 validation/test 为 95.44%/93.88%，
T=5 为 94.96%/93.98%，T=8 为 95.26%/93.91%。实验支持以下结论：

1. Quantile 单独使用没有提高准确率；
2. 单独提高 firing ratio 只有有限收益；
3. Quantile + 约 30% firing ratio 存在明显协同；
4. T=4 的 validation 最高且 MAC 最低，适合作为下一阶段研究 baseline；
5. 三个组合 test 极接近，不能用单 seed 解释 0.1 pct 级差异。

## 时间读出经验

历史 accumulated-membrane 读出在 T=4 对各时刻输出电流形成 4:3:2:1 的 early-time
advantage。一次完全等权 final-membrane 实验使 test 从 91.41% 降至 89.84%，L2 firing
增加 51.84%，有效突触加法增加 20.07%，说明直接完全取消 early-fire advantage 不合适。
但该实验同时改变 temporal weight shape 与 logits scale，不能证明当前 early-time bias 已最优；
温和削弱且保持总权重尺度的结果见 [时间权重记录](records/temporal_weighting_record.md)。

## 架构理解

每个 SNN time step 内执行完整前馈：

```text
input spike(t)
→ Conv1
→ IF1 update
→ Conv2
→ IF2 update
→ Linear
→ output update
```

完成后才进入 `t+1`。因此 T 不是网络完成空间传播所需的深度，而是神经元状态跨时间积分
的次数；卷积提取空间关系，IF membrane 整合不同时刻到达的空间证据。

## 当前固定输入 baseline

本轮后续实验固定 `T=4`、Quantile、约 30% firing ratio、seed=17、
`g_threshold=0.21026152308606838`，复用 training-set quantile boundaries `[0,1,3]`。
既有 beta=1 参考为 validation 95.44%、test 93.88%、352,256 MAC/张、L1 0.2429、
L2 0.7028 fire/IF/张、23,758.61 有效突触加法/张；baseline 不重训。

## 开放假设

尚未在本轮验证：若真实器件的光强→latency 曲线发生单调变形，但仍保持“光强越大、
越早 crossing”，则重新校准 firing threshold 与 quantile boundaries 后，可能无需重新训练整个
SNN。器件迁移、反色输入和 FPGA 均不在本轮范围内。

## 本轮新增知识

固定总权重尺度后，beta=0.5 的 validation 95.66%，高于 beta=1 的 95.44% 和 beta=0.25
的 95.58%；相对 beta=1，其有效突触加法也下降 2.33%。本轮单 seed 结果支持“完全等权失败后，
温和削弱 early-time bias 可行”，但不支持完全取消 early-fire advantage，也不构成统计显著性结论。

以 beta=0.5、lambda=0 为事件阶段基准，lambda=0.01/0.03/0.10 的 validation 损失分别为
0.28/0.12/0.46 pct，均在 0.5 pct 预算内；有效突触加法分别下降 14.45%/36.06%/59.32%。
因此按预注册规则选择 lambda=0.10，得到 validation/test 95.20%/93.60% 和 9,440.04
有效突触加法/张。L1 fire 降 72.27%，L2 降 43.75%，L1 更容易被压缩。lambda=0.03
相对 0.01 同时改善 validation 和事件成本，而 lambda=0.10 用进一步活动下降换取准确率，
形成明确 Pareto trade-off。

当前最合理的研究 baseline 是 T=4、Quantile、约 30% input firing、beta=0.5、lambda=0.10。
它仍只是 seed=17 的方向筛选；多 seed 稳健性、真实器件迁移、量化和 FPGA 证据均未解决，
不能宣布 production champion。

用户随后追加的同配置 seed=17 确认运行得到完全相同的 checkpoint、逐轮非耗时指标和最终结果，
验证了当前软件/硬件环境中的确定性复现；它不增加统计独立样本，不能替代多 seed 验证。

## 最终冻结与三 seed 结果

2026-09-19 冻结 T=4、Quantile、约 30% input firing、beta=0.5、lambda=0.10，不再搜索网络、
T 或超参数。只新增 seed 7/27 两次训练并复用 seed 17 正式结果，三 seed test 为
93.47%/93.60%/93.33%，均值 93.47%、sample std 0.14 pct；Best-Val mean 为 94.97%。
平均 L1/L2 activity 为 0.0691/0.4506 fire/IF/image，平均 hidden spikes 为 301.45/image，
平均 effective synaptic additions 为 9,921.36/image。

matched CNN 的 test mean/std 为 97.37%/0.11 pct，领先 frozen SNN 3.90 pct；两者参数量
9,930/9,872 接近。因此 CNN 的准确率优势明确，SNN 的事件稀疏性是另一维度的代理优势，
在 FPGA 实测前不能据此宣称 SNN 整体或能效胜出。

## 器件曲线迁移

三个预先固定的单调变体只用 training split 重校准约 30% firing threshold 与 T=4 Quantile
boundaries，不更新 SNN 权重。更快 tau=3、更慢 tau=8、基线/增益变化三者的三-checkpoint
test mean 分别为 92.95%、93.21%、93.47%，相对原器件下降 0.52、0.26、0.00 pct。

结果支持 fixed firing ratio + Quantile 对简单单调响应变化具有前端解耦作用，但快速响应造成的
离散 latency 碰撞仍留下小幅损失。证据只来自解析模拟曲线；真实器件噪声、漂移、非单调性和
FPGA 功耗仍未验证。
