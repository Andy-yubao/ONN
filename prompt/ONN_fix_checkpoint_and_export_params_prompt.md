# ONN：固定 RTL Checkpoint 并导出参数

请在当前 `main` 分支上完成本任务。

## 目标

先固定唯一用于后续 RTL 实现的 SNN checkpoint，再导出完整硬件参数。暂时不要编写 RTL。

## 任务

1. 检查 `seed 7 / 17 / 27` 的冻结模型记录与量化结果。
2. 选择一个唯一 checkpoint 作为后续 RTL 使用的 deployment checkpoint。
   - 优先依据 validation、复现一致性和现有冻结记录；
   - 不要仅依据 test accuracy 选择；
   - 在文档中简要说明选择理由。
3. 将该 checkpoint 的部署身份正式记录到 `model/snn/`，后续 RTL 默认只使用这一份参数。
4. 基于现有 INT8 + guard-4 integer reference 导出硬件参数，至少包括：
   - Conv1 INT8 weights
   - Conv2 INT8 weights
   - FC/readout INT8 weights
   - 两层 IF threshold 整数值
   - 各层数据位宽
   - temporal coefficients `[5,4,3,2]`
   - 必要的 scale / quantization metadata
5. 建立清晰的导出目录，例如：

```text
model/snn/export/
├── conv1_weight.mem
├── conv2_weight.mem
├── readout_weight.mem
├── params.svh
└── manifest.json
```

6. `manifest.json` 中记录：
   - seed
   - checkpoint 路径
   - checkpoint SHA256
   - weight shape
   - scale
   - threshold
   - bit width
   - quantization rule
   - export 文件对应关系

7. 增加一个轻量验证脚本或测试，确认：
   - 导出后的权重可无损读回为原 INT8 tensor；
   - shape、数量、范围正确；
   - 导出的 threshold / bit width / temporal coefficients 与当前 `IntegerSNNReference` 一致。

## 完成后

更新必要的 `model/README.md` 或量化记录，明确：

- 已固定唯一 RTL checkpoint；
- 参数导出已完成；
- 下一阶段才进入 RTL；
- 尚未开始 Vivado / synthesis / FPGA 验证。

最后运行相关轻量测试，并报告：
- 最终选择的 seed；
- 导出文件列表；
- 测试结果。
