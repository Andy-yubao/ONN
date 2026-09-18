# `fpga/`：正式模型的硬件实现

`fpga/` 只用于实现 `model/` 中已正式选定的 champion。当前没有新的 RTL、仿真工程、
综合工程或 bitstream。

AC620 / Cyclone IV 属于 legacy platform；当前新的优先候选目标板是 Basys3 / Artix-7。
在 champion 模型、数据位宽、状态存储和资源预算冻结前，不创建最终 Vivado 工程，也不
宣称硬件迁移已经完成。
