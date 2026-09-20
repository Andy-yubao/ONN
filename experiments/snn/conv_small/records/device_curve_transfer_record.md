# Frozen SNN 器件曲线迁移实验

日期：2026-09-19。状态：三种器件变体、三个冻结 checkpoint，共 9 次新增 test inference
均已完成。没有训练或 fine-tune SNN 权重；原器件行复用每个 seed 已有的唯一 final test。

## 方法

固定模型为 [frozen_model.md](../frozen_model.md)。对每个新器件只使用 55k training split：

1. 选择固定 conductance threshold，使 24 步窗口内的平均 input firing ratio 接近 30%；
2. 由 training firing latency histogram 计算 `T=4` Quantile boundaries；
3. 固定 encoder，validation/test 不参与校准；
4. 直接加载 seed 7/17/27 frozen checkpoint 做 test inference。

三个变体不是参数搜索：A 改为更快响应，B 改为更慢响应，C 同时提高基线并降低增益。对
`p∈[0,1]` 的 1,001 点网格检查确认三者 conductance 均随光强和时间非减，且有效 crossing
latency 随光强增加而非增，曲线有效。

## 结果

| Device | G0 | alpha | tau | Calibrated threshold | Actual firing ratio | Quantile boundaries | Seed 7 Test | Seed 17 Test | Seed 27 Test | Mean | Drop vs original |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Original | 0.10 | 0.90 | 5.0 | 0.210261523 | 30.0866% | `[0,1,3]` | 93.47% | 93.60% | 93.33% | 93.47% | — |
| Faster response | 0.10 | 0.90 | 3.0 | 0.211139182 | 30.0866% | `[0,1,2]` | 92.85% | 93.38% | 92.61% | 92.95% | 0.52 pct |
| Slower response | 0.10 | 0.90 | 8.0 | 0.205641326 | 30.0866% | `[1,2,4]` | 93.29% | 93.36% | 92.98% | 93.21% | 0.26 pct |
| Baseline/gain change | 0.16 | 0.70 | 5.0 | 0.245758962 | 30.0866% | `[0,1,3]` | 93.47% | 93.60% | 93.33% | 93.47% | 0.00 pct |

完整校准 histogram、逐 checkpoint accuracy、activity、checkpoint hash 和曲线检查见
[`summary.json`](../results/device_curve_transfer/summary.json)。执行入口为
[`device_curve_transfer.py`](../device_curve_transfer.py)。

## 结论与边界

1. **准确率基本保持。** 三种变体的最差 mean drop 为 0.52 pct，且所有变体仍在
   92.95%–93.47% mean test accuracy 范围内。
2. **基线/增益变化最容易迁移。** 当 tau 不变时，重校准 threshold 后 source latency 与
   Quantile code 完全恢复，三个 checkpoint 的 accuracy 逐项不变。
3. **更快响应影响最大。** tau=3 使离散 source latency 更强地压缩，boundary 变为
   `[0,1,2]`；虽经 Quantile 重校准，仍有 0.52 pct mean drop。tau=8 的下降较小。
4. **支持有限的解耦作用。** fixed firing ratio 保持事件覆盖率，Quantile boundaries 吸收了
   大部分单调 latency 变形；但离散时间的碰撞/舍入不能完全逆转，因此不是严格不变性。
5. **在本模拟范围内支持**“换器件 → 重校准 encoder → 复用同一套 SNN 权重”。该证据仅
   覆盖三个简单、解析、单调器件模型；尚未覆盖真实器件噪声、非单调性、漂移、测量误差或
   FPGA 实测功耗，不能替代真实器件验证。

