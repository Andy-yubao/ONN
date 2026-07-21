# Channel Activity Analysis Summary

Statistics are based on post-activation outputs
(epsilon = 1e-06).

## Layer: stem_output
  Shape: [8, 16, 28, 28]
  Samples: 5000  |  Batches: 40
  Mean abs: 0.327389 ± 0.149367
  Most active channels:    [11, 13, 15, 2, 8]
  Least active channels:   [14, 10, 3, 7, 9]
  Channels mean_abs < 0.01: 0/16
  Channels near_zero_ratio > 0.9: 0/16

## Layer: conv2_output
  Shape: [8, 32, 14, 14]
  Samples: 5000  |  Batches: 40
  Mean abs: 0.231656 ± 0.082190
  Most active channels:    [30, 29, 3, 17, 5]
  Least active channels:   [11, 9, 1, 28, 18]
  Channels mean_abs < 0.01: 0/32
  Channels near_zero_ratio > 0.9: 0/32

## Layer: conv3_output
  Shape: [8, 32, 7, 7]
  Samples: 5000  |  Batches: 40
  Mean abs: 1.101538 ± 0.081791
  Most active channels:    [19, 11, 8, 4, 7]
  Least active channels:   [23, 21, 28, 22, 14]
  Channels mean_abs < 0.01: 0/32
  Channels near_zero_ratio > 0.9: 0/32


## Limitations

- Mean absolute values are **not directly comparable across layers** because they depend on the preceding BatchNorm scale and the layer's own weight distribution.
- A low active-ratio channel may still be useful for discriminating a small number of classes.  Do not prune based solely on activity.
- These statistics describe the validation-set behaviour of the *trained* model.  Activity patterns can change during training.