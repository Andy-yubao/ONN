# 论文笔记 05：FPGA 全连接 ANN/MNIST 示例

## 论文信息

- **标题**：Detecting Handwritten digits using fully connected ANN in FPGA
- **论文编号**：P5
- **证据等级**：C（FPGA 实现参考——低证据等级）

## 纳入原因

1. 完整流程：Python 训练 → 权重导出 → 定点表示 → Verilog 推理 → testbench
2. 流程结构可参考

## 核心内容

### 流程
1. Keras/TensorFlow 训练（全连接 784-30-30-10-10）
2. 权重导出为二进制文件
3. 定点量化
4. Verilog 推理模块 + BRAM 存储
5. testbench 验证

### 实现
- Zynq 7000（不是 Cyclone IV）
- 15 BRAM, 6370 LUT, 5369 FF, 160 DSP

## 方法学缺陷（批判性）
1. **层数描述不一致**：文内不同位置描述 3 和 4 个隐藏层
2. **激活函数错误**：声称 sigmoid"允许负值"——错误（sigmoid 输出 [0,1]）
3. **功耗数据不可靠**：仅以百分比展示，无绝对值
4. **GPU 对比不严谨**：训练样本数量不同，宣称 FPGA 优于 GPU 的依据不足

## 可复用
- 训练→导出→FPGA 的工作流结构
- testbench 方法

## 不可移植
- 所有性能数据不能引用
- "FPGA 优于 GPU"结论不能引用
- 98% 准确率不能作项目性能承诺

## 与 4×4 关系
- 流程参考（训练→导出→部署）
- 4×4 需更小网络（16 输入而非 784）

## 关键章节
- Sec. II: 训练和量化
- Sec. III: 硬件实现
- Sec. IV: 结果（批判性阅读）
