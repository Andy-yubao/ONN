# CNN / SNN 比较

已完成的软件准确率比较：

- [8×8 冻结 SNN 与 matched CNN 三 seed 对照](../snn/conv_small/records/final_three_seed_record.md)；
- [8×8 IF/LIF 泄露消融](../snn/lif_if/README.md)；
- [28×28 同参数量 CNN/SNN 三 seed 对照](software_28/README.md)。

28×28 的 IF 结果是在已接触同一测试集后进行的事后复核，解释范围见实验记录。

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
