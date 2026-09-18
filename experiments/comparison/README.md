# CNN / SNN 公平比较（待开展）

最终比较至少报告：

```text
Accuracy
Parameters
Model storage
Dense-equivalent operations
Effective event/synaptic operations
Activation/state memory
Hardware mapping difficulty
Expected FPGA BRAM/DSP/LUT pressure
```

不能只比较参数量，不能把 GPU inference time 当作 FPGA 性能结论，也不能把 dense MAC
与 event addition 当作同一种物理工作量。最终报告要区分“精度比较”和“硬件代价比较”，
当前不填最终胜负结论。
