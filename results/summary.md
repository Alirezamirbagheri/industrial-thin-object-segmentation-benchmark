# Final Benchmark Summary

| Model | Wire Dice | Boundary F1 | Native latency | FPS |
|---|---:|---:|---:|---:|
| U-Net | 0.889 | 0.847 | 75.7 ms | 13.2 |
| YOLO26s-Sem | 0.886 | 0.835 | 24.5 ms | 40.8 |
| PIDNet-S | 0.881 | 0.820 | 24.2 ms | 41.3 |
| U-Net++ | 0.879 | 0.807 | 176.6 ms | 5.7 |
| SegFormer-B0 | 0.878 | 0.807 | 220.9 ms | 4.5 |
| DeepLabV3+ | 0.877 | 0.801 | 52.9 ms | 18.9 |
| HRNet-W18-Small-v2 | 0.863 | 0.763 | 96.6 ms | 10.3 |

U-Net provides the strongest segmentation quality in the final test, while YOLO26s-Sem provides the strongest accuracy/speed trade-off for deployment-oriented use.
