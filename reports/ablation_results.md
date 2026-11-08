| Configuration | Params | Song acc. (soft vote) | Slice acc. (no vote) | Macro F1 | Best epoch |
|---|---:|---:|---:|---:|---:|
| Full Attention-CRNN (CNN + BiGRU + attention + SpecAugment, soft vote) | 1,017,290 | 0.870 | 0.813 | 0.869 | 25 |
| w/o temporal attention (mean pooling over time) | 984,266 | 0.865 | 0.800 | 0.867 | 19 |
| w/o BiGRU (per-step linear projection) | 490,186 | 0.870 | 0.810 | 0.870 | 28 |
| w/o SpecAugment | 1,017,290 | 0.860 | 0.813 | 0.862 | 27 |
| Attention-CNN on full 30 s clip (no slicing, no BiGRU) | 407,883 | 0.830 | – | 0.827 | 23 |
