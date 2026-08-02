# 实验记录

## 目录说明

```
experiments/
├─ README.md                       # 本文件
├─ templates/                      # 实验记录模板
│  └─ device_characterization.md   # 器件表征实验模板（路线 A）
├─ model_baselines/                # 模型基线实验
│  ├─ m1_baselines/                # M1-B 初始基线（单种子，含已废弃 pre-activation 分析）
│  └─ m1_baseline_audit/           # M1-B.1 审计与可复现性加固（多种子 post-activation）
├─ model_compaction/               # M2 compact 模型探索
│  └─ m2_compact_models/
└─ model_deployment/               # FPGA 部署链路实验
   ├─ baseline_cnn_bn_fusion/      # BN 融合验证
   ├─ baseline_cnn_int8_ptq/       # INT8 PTQ 可行性评估
   └─ baseline_cnn_int8_reference/ # 纯整数参考模型验证
```

## 实验记录规范

每次实验创建一份记录文件，命名格式：`YYYY-MM-DD_实验名称.md`

记录内容包括：
- 实验目的
- 日期和参与者
- 使用硬件与软件版本
- 实验步骤
- 参数配置
- 结果（数据、图表）
- 异常现象
- 结论
- 下一步

> 重点记录"失败实验"。成功结果可以重新得到，但失败的原因很容易被遗忘。

## 实验数值的唯一来源

准确率、量化后精度和多种子统计以 `experiments/**/README.md` 与结果文件（JSON/CSV）为准，项目汇总文档只做摘要并链接，不复制数值。
