# BaselineCNN 硬件数据格式（scheme A 纯整数参考模型导出）

> 适用对象：`fpga/baseline_cnn/` 参数包、ModelSim 黄金测试向量与 RTL。
> 唯一数值标准：冻结的 `Int8Reference`（`model/onn_model/int8_reference.py`）及其
> `candidate_quant_config.json`（方案 A，per-tensor 权重）。本文件不推导任何量化公式。
> 当前 RTL 状态：算术 smoke、stem 卷积引擎、流式 MaxPool 与 stem+pool1 集成
> 阶段已落地——
> `requantize_u8.v` / `gap_div49.v` 通过 Questa 黄金向量验证；`stem_conv_serial.v`
> （单 MAC 串行 stem 卷积，冻结设计见 `docs/rtl_microarchitecture.md`）通过 digit8
> 黄金 trace 全量验证并在 EP4CE10F17C8 上完成综合/Fitter（新增 `STORE_OUTPUT_RAM`
> 参数，`=0` 时集成工程不实例化完整 stem 输出 RAM）；`maxpool2x2_stream.v`
> （流式 2×2 MaxPool，§9）通过 `pool1_q` 两遍 golden 验证（连续/空拍）并在
> EP4CE10F17C8 上完成综合/Fitter；`stem_pool1_pipeline.v`（§10 集成核心）实现
> `input_q → stem → requant → 流式 MaxPool → pool1 RAM`，两遍推理逐位一致并在
> EP4CE10F17C8 上完成综合/Fitter（无 UART，无 conv2/conv3）。

## 1. 整数类型与补码表示

| 数据 | 位宽 | 符号 | 表示 | 范围 |
|---|---|---|---|---|
| 权重（weight） | 8 | signed | 二进制补码 | `[-127, 127]`，zero_point=0 |
| 输入（input_q） | 8 | signed | 二进制补码 | `[-128, 127]`，zero_point=0 |
| ReLU 输出（stem_q/conv2_q/conv3_q/gap_q） | 8 | unsigned | 位模式 `00`–`FF` | `[0, 255]`，zero_point=0 |
| bias | 32 | signed | 二进制补码 | `[-2³¹, 2³¹-1]` |
| 累加器（conv1_acc/…/fc_acc） | 32 | signed | 二进制补码 | `[-2³¹, 2³¹-1]` |
| MaxPool 输出（pool1_q/pool2_q） | 8 | unsigned | 位模式 | `[0, 255]` |

说明：

* 权重与输入是有符号补码；ReLU 之后全是无符号 `[0,255]`，在文件里就是 `00`–`FF`
  的 8 位位模式，读回时按无符号解释。
* bias 与累加器是 32 位有符号补码，每行 8 位十六进制（如 `FFFFFF4C` = −180）。
* 任何地方都不允许整数回绕：超出范围必须**饱和（clamp）**。

## 2. `.mem` 与 `.mif` 读取方式

每个权重 / bias / 特征图同时给两种格式，内容等价：

### `.mem`（ModelSim `$readmemh`）

每行一个十六进制值，地址从 0 连续递增：

```text
$readmemh("stem_weight.mem", mem);   // mem: reg [7:0] mem[0:143];
```

* 权重文件：每行 2 位十六进制；bias 文件：每行 8 位十六进制。
* 文件内没有注释、没有 `@地址` 指令，`$readmemh` 直接按行序装入 `mem[0]`、`mem[1]`…
* 若用 `reg signed [7:0]` 接收，负权重的 `B6` 会自动解释为 −74（Verilog 按位向量加载，
  有符号性由后续算术决定）。

### `.mif`（Quartus ROM 初始化）

标准 MIF 文件，头部声明 `WIDTH`/`DEPTH`，`CONTENT BEGIN … END` 内每行一条
`地址 : 数据;`：

```text
WIDTH = 8;
DEPTH = 144;
ADDRESS_RADIX = DEC;
DATA_RADIX = HEX;
CONTENT
BEGIN
0 : B6;
1 : CA;
...
END;
```

地址为十进制、数据为十六进制，地址必须连续（验证脚本会检查）。

## 3. 地址公式（展开顺序）

### 卷积权重 OIHW

```text
addr = ((out_channel * Cin + in_channel) * Kh + ky) * Kw + kx
```

即 PyTorch `[Cout, Cin, Kh, Kw]` 张量的行主序展开。例如 `stem_weight`（16×1×3×3=144）：
第 0 个输出通道的第 0 输入通道的 3×3 内核占用 `addr 0..8`。

### 全连接权重 OI

```text
addr = out_feature * in_features + in_feature_index
```

即 `[10, 32]` 张量的行主序展开：第 0 类别的 32 个权重在 `addr 0..31`。

### 特征图 CHW

```text
addr = (channel * height + y) * width + x
```

所有 `*.mem` 特征图（`input_q`、`stem_q`、`conv1_acc` 等）均按此展开。
例：`stem_q`（16×28×28）`channel=0, y=0, x=0` → `addr 0`；`channel=1, y=0, x=0` →
`addr 784`。

### 每层几何与元素数

| 文件 | shape | 元素数 | 布局 |
|---|---|---|---|
| `stem_weight` | 16×1×3×3 | 144 | OIHW |
| `conv2_weight` | 32×16×3×3 | 4608 | OIHW |
| `conv3_weight` | 32×32×3×3 | 9216 | OIHW |
| `fc_weight` | 10×32 | 320 | OI |
| `stem_bias` / `conv2_bias` / `conv3_bias` / `fc_bias` | 16 / 32 / 32 / 10 | — | 线性 |
| `input_q` | 1×28×28 | 784 | CHW |
| `conv1_acc` | 16×28×28 | 12544 | CHW（INT32） |
| `stem_q` / `pool1_q` | 16×28×28 / 16×14×14 | 12544 / 3136 | CHW |
| `conv2_acc` / `conv2_q` / `pool2_q` | 32×14×14 / 32×14×14 / 32×7×7 | 6272 / 6272 / 1568 | CHW |
| `conv3_acc` / `conv3_q` | 32×7×7 / 32×7×7 | 1568 / 1568 | CHW |
| `gap_q` | 32 | 32 | 线性 |
| `fc_acc` | 10 | 10 | 线性（INT32） |

## 4. 输入约定（PC 预处理，FPGA 不做浮点）

**FPGA 第一版不负责浮点归一化。** PC 侧完成全部浮点工作后，把已经量化的 `input_q`
逐字节发给 FPGA：

```text
原图像素 p (0..255)
→ x_norm = (p/255 - 0.1307) / 0.3081        # MNIST ToTensor + Normalize，PC 侧浮点
→ input_q = round_half_away_from_zero(x_norm / s_in)，饱和到 [-128, 127]
→ 按 CHW 逐字节发送
```

* `s_in = 0.02221643079922894`（`manifest.json` → `network.input_scale`）；
* 舍入规则统一为 **round-half-away-from-zero**（0.5→1，−0.5→−1），**禁止** round-half-to-even；
* FPGA 收到的就是有符号 INT8 的位模式，直接进卷积 MAC。

## 5. 各层计算规则（FPGA 须与整数参考逐位一致）

### 5.1 卷积 + bias（INT32 累加）

```text
acc[y][x] = Σ_{ic,ky,kx} input_q[...] * weight[...] + bias
```

* 权重有符号 INT8，输入有符号 INT8（第一层）或无符号 UINT8（后续层）；
* 乘累加在 **INT32** 内进行；若真实和超出 INT32（本模型实测不会，见 §7），必须**饱和**
  而非回绕；
* bias 在 MAC 求和后一次性加入。

### 5.2 定点 requantization → UINT8 ReLU（仅三个卷积层）

顺序固定为：

```text
q = saturate_uint8( round_half_away_from_zero( (acc * MULTIPLIER) >> SHIFT ) )
```

1. `acc * MULTIPLIER` 用 **INT64** 中间值（乘积 ~2³¹×2³¹ 需要 62 位）；
2. **round-half-away-from-zero** 右移 `SHIFT` 位（不能直接算术右移，见 §6）；
3. 结果**饱和**到 UINT8 `[0,255]`。

各层常数（`manifest.json` → `network.layers.*.requant`，或 `baseline_cnn_params.vh`）：

| 层 | MULTIPLIER | SHIFT |
|---|---|---|
| stem_conv | 2,056,884,242（`32'h7A999012`） | 38 |
| conv2 | 1,097,020,857（`32'h416335B9`） | 38 |
| conv3 | 1,298,974,956（`32'h4D6CC8EC`） | 38 |

* multiplier 均为正、小于 `2³¹`（signed 31 位量级），可在 32 位有符号寄存器中表示；
* 乘法用**有符号**运算（`acc` 可负）；移位按**有符号**量做（§6）。

### 5.3 MaxPool（2×2 stride 2，保持 scale）

```text
pool_q[y][x] = max( q[2y][2x], q[2y][2x+1], q[2y+1][2x], q[2y+1][2x+1] )
```

* 直接对 UINT8 取 max，**不重新量化**，scale 保持不变；
* 池化输出仍是无符号 `[0,255]`。

### 5.4 GAP（整数求和 ÷49）

```text
gap_sum[c] = Σ_{y,x} conv3_q[c][y][x]        # 对每个通道 7×7 求和
gap_q[c]   = round_half_away_from_zero(gap_sum[c] / 49)，饱和到 [0,255]
```

* 是**整数除法** `sum/49` 并 round-half-away-from-zero（如 74→2、25→1、24→0），
  不是浮点均值再量化；
* `gap_q` 保持在 **conv3 的单位**（scale = s_conv3_relu），因此 fc 的输入 activation
  scale 是 `s_conv3_relu`，fc bias 按 `s_conv3_relu × sw_fc` 量化；
* 这一点与 fake-quant 对照（按 s_pool 重量化）是**冻结的设计差异**，不是错误。

### 5.5 全连接 → 预测

```text
fc_acc[o] = Σ_f gap_q[f] * fc_weight[o][f] + fc_bias[o]      # INT32 累加
prediction = argmax(fc_acc)                                    # 直接对 INT32 累加器取最大
```

* fc 输入 `gap_q` 是无符号 UINT8（0..255），权重是有符号 INT8；
* **没有**对 fc 做任何 requantization / dequantization —— 直接对 INT32 accumulator
  `argmax`；
* 累加同样在 INT32 内完成，超界饱和、不回绕。

## 6. Verilog 中必须使用 signed 运算的位置

1. **乘法器**：`acc * MULTIPLIER` 必须 `signed`（`acc` 是 signed INT32，multiplier 是
   signed 正数；结果是 62 位 signed）。
2. **右移**：requant 右移必须按**有符号量级** round-half-away-from-zero —— 对负数先取
   绝对值、加 `2^(SHIFT-1)`、右移、再回符号；不能用 Verilog 的 `>>>`（它是向下取整，
   与 round-half-away-from-zero 不同）。RTL 阶段可用
   `q = (|acc*mult| + 2^(SHIFT-1)) >> SHIFT`，再按原符号变号。
3. **卷积 / fc 累加**：权重、bias、累加器都是 signed；但 ReLU 前的偏置求和结果可负。
4. **比较与饱和**：`clamp` 到 UINT8 / INT32 上下界用有符号比较。

## 7. 累加器安全余量（10,000 测试集实测）

`integer_reference_results.json`（既有实验结果）记录的最大绝对值：stem_conv 49,552、
conv2 90,257、conv3 54,445、fc 39,153，距 INT32 上下界均有约 2.1×10⁹ 余量，溢出计数为 0。
即正常输入不会触发饱和；饱和逻辑只是防错回绕的最后一道防线。

## 8. 文件索引

* `params/manifest.json`：全部文件哈希、位宽、元素数、scale、requant 常数、布局；
* `params/baseline_cnn_params.vh`：标量参数头（通道/宽高、multiplier/shift、GAP_DIVISOR、
  权重与 bias depth）；
* `params/checksums.sha256`：受控文件哈希；
* `sim/vectors/smoke/smoke_manifest.json`：每数字一个样本（原测试集索引、标签、预测）；
* `sim/vectors/golden_trace/trace_manifest.json`：单样本逐层节点的 shape/dtype/numel/min/max/SHA256。
