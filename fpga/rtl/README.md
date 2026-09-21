# SNN RTL 数值与接口契约

当前目录已完成 Phase 1–4 的仿真级完整推理链路：

```text
64-bit timestep spike → Conv1 → IF1 → Conv2 → IF2
→ 512→10 readout → temporal weighted score → argmax
```

部署顶层已经切换为 spike-driven sparse 架构；原 dense 模块保留为黄金基线。尚未冻结并行度，
未创建最终 Vivado 工程，也未执行 synthesis / implementation。

## 冻结数值契约

`onn_snn_params_pkg` 只包含一次
[`../../model/snn/export/params.svh`](../../model/snn/export/params.svh)，各模块从 package
导入冻结常量。三个权重 ROM 分别直接读取同目录的 `.mem` 文件。完整部署契约如下：

| 项目 | 契约 |
|---|---|
| weight | INT8 signed |
| Conv1 current / LIF1 membrane | INT16 signed / INT18 signed |
| Conv2 current / LIF2 membrane | INT20 signed / INT22 signed |
| readout current / weighted score | INT17 signed / INT21 signed |
| LIF1 / LIF2 threshold | 939 / 1715 |
| temporal coefficients | `[5,4,3,2]` |
| threshold compare | 严格 `>` |
| reset | spike 后只减一次 threshold |
| overflow | 每个规定边界做 signed saturation，不允许 wrap-around |

IF 顺序固定为 current clamp、`membrane + current` clamp、严格阈值比较、减一次阈值、
post-reset clamp。所有加法先显式符号扩展一位，再进入 clamp。

## 顶层接口与时序

`onn_snn_core` 只在 `ready=1` 时接收 `step_valid`。`spike_in[y*8+x]` 对应输入像素
`[y,x]`。新样本第一个 timestep 必须置 `frame_start=1`，后续三个 timestep 置零。
busy 期间外部 `step_valid` 不会改变已接收的输入或内部状态。每个 timestep 完成时输出
`step_done`；第四步完成时同时输出 `frame_done` 和 `class_out`。

顶层保留 L1、L2 和 readout 的流式调试口，用于逐层、逐元素对拍。最终资源架构冻结时可根据
调试需求决定是否保留这些端口。顶层还输出每个 timestep 和整帧的实际突触加法数、计算
周期数，用于验证工作量确实随 spike 数变化。

## Spike-driven Conv1 + IF1

`onn_sparse_conv1_if1` 用 64-bit priority encoder 逐个提取有效输入事件。每个事件只把
对应 3×3 kernel 权重散射累加到受影响的 16 个输出 channel；零 spike 不进入累加循环。
一个输入事件最多占用 `16×9=144` 个 scatter slot，边界外位置不会计入突触加法数。

权重地址仍为
`channel*9 + kernel_y*3 + kernel_x`，与导出的 PyTorch contiguous `[16,1,3,3]`
顺序一致。事件耗尽后执行一次 1024 项 IF1 barrier sweep：读出累计 current、更新膜状态、
产生 spike，同时清零本 timestep current。这个 sweep 是状态同步，不是 dense 突触计算。

结果以流方式输出完整 `16×8×8` tensor。`result_index=channel*64+y*8+x`；
`result_valid` 时同步给出 current、integrated membrane、post-reset membrane 和 spike。
最后一个元素同时置 `step_done`。

## Spike-driven Conv2 + IF2

`onn_sparse_conv2_if2` 以 32-bit group 扫描加 priority encoder 从 `[16,8,8]` bitmap
提取 L1 spike。每个事件枚举 output channel 和 3×3 kernel，并只在 stride=2 坐标关系
成立时写入 `[32,4,4]` current accumulator。零 spike 不执行权重累加。权重地址为：

```text
output_channel*144 + input_channel*9 + kernel_y*3 + kernel_x
```

事件耗尽后执行 512 项 IF2 barrier sweep。raw accumulator 为 INT16，左移 4 位后生成
INT20 current，随后由阈值 1715 的 INT22 IF2 更新状态。

## Spike-driven Readout 与 argmax

`onn_sparse_readout` 每提取一个 L2 spike，只向十个类别各累加一次对应权重。因此每个事件
固定产生 10 次突触加法，而不是扫描 `10×512` 个连接。事件耗尽后用 `[5,4,3,2]`
更新十个 INT21 score。第四步完成 signed argmax；分数相同保留较小类别编号。

## Sparse 与 dense 基线

以下原始模块继续保留并参加独立 testbench，用作易审计的 dense 黄金 RTL：

- `onn_conv1_if1`；
- `onn_conv2_if2`；
- `onn_readout`；
- `onn_snn_core_dense`。

部署顶层 `onn_snn_core` 使用三个 `onn_sparse_*` 模块。dense 基线每帧固定扫描 352,256 个
连接槽位；sparse 顶层的实际加法数和延迟依赖事件数量。当前端到端向量中的两个真实
MNIST validation 样本结果为：

| 样本 | sparse 突触加法 | sparse 四步计算周期 | dense 约 353.8k 周期的比值 |
|---|---:|---:|---:|
| train index 55002 | 9,130 | 31,603 | 11.2× fewer cycles |
| train index 55003 | 8,234 | 28,360 | 12.5× fewer cycles |

周期统计包含首步状态清零、事件提取、scatter slot、完整 IF barrier 和 readout 更新，
不包含少量顶层交接周期。当前仍是一条 scatter lane；后续并行度需由综合资源和 BRAM
端口约束决定。
