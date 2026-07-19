# 神经网络模型

## 目录说明

```
model/
├─ README.md          # 本文件
├─ training/          # 训练代码
├─ deployment/        # 部署代码与导出脚本
├─ configs/           # 训练与模型配置文件
└─ evaluation/        # 评估代码与结果
```

## 工作流程

### B 轨：28×28 CNN 数字推理（主基线）

1. 在 `training/` 中开发普通 CNN（Adam）训练脚本
2. 配置文件和超参数保存在 `configs/`
3. 在 `evaluation/` 中评估浮点基线准确率
4. 在 `training/` 中实现定点/量化参考模型
5. 导出定点权重并转换格式到 `deployment/`，供 FPGA 使用

### C 轨：器件曲线优化探索（辅助，不阻塞 B 轨）

1. （基线建立后）基于论文器件曲线实现自定义优化器
2. 与普通 Adam 基线对照比较
3. 用未来实测曲线替换论文参数

> 普通 Adam CNN 必须作为 FPGA 推理基线和器件曲线方法对照。不得预设器件曲线优化器一定优于 Adam。

## 依赖

- Python 3.x
- PyTorch
- NumPy
- Matplotlib（可视化）
