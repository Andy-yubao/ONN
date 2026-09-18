# 8×8 CNN baseline（待开展）

本目录只定义下一轮公平实验要求，不训练 CNN，也不预先设计最终架构。

目标是建立一个与当前轻量 SNN 参数量相近的 8×8 CNN baseline。当前参考 SNN 约
9,872 parameters，CNN 目标规模建议约 8k–12k；最终结构由实验决定。

未来 CNN 与 SNN 至少统一：

- MNIST 8×8 输入与相同 resize/preprocessing 原则；
- 55k train / 5k validation / 10k test；
- test set 只用于最终一次评价；
- 相同 seed policy；
- comparable parameter scale；
- 不用 test accuracy 做 architecture selection。
