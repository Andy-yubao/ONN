# Channel Activity Analysis Summary

Statistics are based on post-activation outputs
(epsilon = 1e-06).

## Layer: stem_output
  Shape: [8, 16, 28, 28]
  Samples: 5000  |  Batches: 40
  Mean abs: 0.325516 ± 0.061511
  Most active channels:    [4, 2, 15, 0, 8]
  Least active channels:   [6, 3, 11, 5, 12]
  Channels mean_abs < 0.01: 0/16
  Channels near_zero_ratio > 0.9: 0/16

## Layer: stage1.block0.output
  Shape: [8, 16, 28, 28]
  Samples: 5000  |  Batches: 40
  Mean abs: 0.554367 ± 0.100918
  Most active channels:    [15, 2, 4, 0, 6]
  Least active channels:   [3, 13, 7, 12, 11]
  Channels mean_abs < 0.01: 0/16
  Channels near_zero_ratio > 0.9: 0/16

## Layer: stage1.block1.output
  Shape: [8, 16, 28, 28]
  Samples: 5000  |  Batches: 40
  Mean abs: 0.811250 ± 0.126610
  Most active channels:    [15, 2, 0, 6, 14]
  Least active channels:   [13, 1, 11, 7, 9]
  Channels mean_abs < 0.01: 0/16
  Channels near_zero_ratio > 0.9: 0/16

## Layer: stage2.block0.output
  Shape: [8, 32, 14, 14]
  Samples: 5000  |  Batches: 40
  Mean abs: 1.011350 ± 0.101969
  Most active channels:    [12, 0, 2, 18, 9]
  Least active channels:   [16, 22, 26, 1, 17]
  Channels mean_abs < 0.01: 0/32
  Channels near_zero_ratio > 0.9: 0/32

## Layer: stage2.block1.output
  Shape: [8, 32, 14, 14]
  Samples: 5000  |  Batches: 40
  Mean abs: 1.744600 ± 0.121599
  Most active channels:    [12, 2, 27, 10, 9]
  Least active channels:   [26, 16, 22, 1, 17]
  Channels mean_abs < 0.01: 0/32
  Channels near_zero_ratio > 0.9: 0/32


## Limitations

- Mean absolute values are **not directly comparable across layers** because they depend on the preceding BatchNorm scale and the layer's own weight distribution.
- A low active-ratio channel may still be useful for discriminating a small number of classes.  Do not prune based solely on activity.
- These statistics describe the validation-set behaviour of the *trained* model.  Activity patterns can change during training.