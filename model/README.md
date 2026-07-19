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

1. 在 `training/` 中开发 CNN 训练脚本
2. 基于器件电导特性在 `training/` 中实现自定义优化器
3. 配置文件和超参数保存在 `configs/`
4. 训练完成后在 `evaluation/` 中评估性能
5. 导出权重并转换格式到 `deployment/`，供 FPGA 使用

## 依赖

- Python 3.x
- PyTorch
- NumPy
- Matplotlib（可视化）
