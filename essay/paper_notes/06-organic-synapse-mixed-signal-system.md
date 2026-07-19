# 论文笔记 06：有机突触 32×32 混合信号系统（含 SI）

## 论文信息

- **标题**：An ultrasmall organic synapse for neuromorphic computing
- **作者**：Liu 等（上海交通大学）
- **期刊**：Nature Communications (2023)
- **论文编号**：P6
- **证据等级**：B（外部系统参考）

## 纳入原因

1. 完整的混合信号系统（DAC + MUX + TIA + ADC + FPGA）
2. 与项目同系列的 Cyclone IV + Quartus
3. 行列寻址方案
4. VMM 实现和信号链
5. 封装和 PCB 设计参考

## 核心内容

### 阵列
- 32×32 有机忆阻交叉阵列
- 1S1R（PEDOT:PSS:Ag NP 选择器）
- 50 nm 线宽，85 nm 间距

### 混合信号系统（SI Sec. 8）
| 组件 | 型号 | 功能 |
|------|------|------|
| DAC | DAC5578 (8-bit) | 行/列电压生成 |
| MUX | MUX508IPWR | 列选择 |
| TIA | LTC6268 | 电流→电压 |
| ADC | ADC128S102 (12-bit) | 数字化 |
| FPGA | Cyclone IV | 控制 + 数字处理 |
| 工具链 | Quartus II | 编译 |

### 读写
- 写入：−0.45 ~ −0.76 V 脉冲，100 µs
- 读取：0.1 V
- 电导状态：32 级量子化（G0 = 77.5 µS）
- 开关速度：~21 ns SET, 115 ns RESET

## 可复用设计
- 行列电压寻址 V_device = V_row − V_col
- VMM 信号链架构
- FPGA 控制方案

## 不可移植
- 有机忆阻 vs 光学 GDY/WSe₂
- **所有元件选型是外部实例**（DAC5578 等）
- 不能假设 4×4 是交叉阵列
- 量子化电导 G0 是 Ag 细丝特性

## 与 4×4 关系
- 信号链蓝本（简化版：4×4 只需 4 通道 MUX + TIA）
- Q 工具链验证（Cyclone IV + Quartus）

## 关键图表
- Fig. 1–3: 器件特性
- Fig. 4: 32×32 阵列
- SI Fig. 22: 混合信号系统框图（核心参考）
- SI Fig. 28: STDP 比较器 RTL
