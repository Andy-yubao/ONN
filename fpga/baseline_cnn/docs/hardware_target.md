# BaselineCNN 硬件目标（AC620 / EP4CE10F17C8）

> 本文档冻结 BaselineCNN 的 FPGA 硬件目标与当前板级状态。
> 数值协议见 `docs/data_format.md`；参数包见 `params/manifest.json`（两者均与板卡无关）。

## 1. 硬件目标（已冻结）

| 项 | 值 |
|---|---|
| Board | AC620（小梅哥） |
| FPGA part | EP4CE10F17C8 |
| Family | Cyclone IV E |
| Package | F17 |
| Speed grade | C8 |
| Quartus edition | Lite |
| Quartus installation | `D:\tools\altera_lite\25.1std` |

## 2. 冻结状态

### 已冻结（Frozen）

- 开发板：AC620
- FPGA 器件：EP4CE10F17C8（Cyclone IV E）
- 模型参数和数值协议：见 `docs/data_format.md`（Int8Reference，scheme A）

### 尚待冻结（To be frozen）

- 板载主时钟频率和引脚
- 复位引脚和有效电平
- UART RX/TX 引脚
- 调试 LED 引脚
- 最终顶层端口

> ⚠ 未获得可靠板卡引脚表前，**不得编造引脚约束**。

## 3. 微架构方向（当前拟采用）

- 一套共享 MAC 计算引擎
- 三个卷积层分时复用
- 第一版优先正确性和低资源占用
- 第一版 UART 以 784 字节 `input_q` 作为输入
