# Industrial Thin-Object Segmentation Benchmark

Benchmark of deep-learning segmentation architectures for thin and deformable industrial structures, developed around a wire-harness perception task.

## Highlights

- Common evaluation protocol across multiple semantic-segmentation architectures
- Pixel metrics: Dice, IoU, precision, recall and F1
- Geometry-aware metrics: boundary F1, clDice and HD95
- Deployment-oriented runtime comparison at crop and native image resolution
- Public-safe training/model/metric code with machine-specific paths removed

## Final test results

| Model | Wire Dice | Boundary F1 | Native latency (ms) | FPS | Parameters |
|---|---:|---:|---:|---:|---:|
| U-Net | **0.889** | **0.847** | 75.7 | 13.2 | 24.4M |
| YOLO26s-Sem | 0.886 | 0.835 | 24.5 | 40.8 | 6.5M |
| U-Net++ | 0.879 | 0.807 | 176.6 | 5.7 | 26.1M |
| SegFormer-B0 | 0.878 | 0.807 | 220.9 | 4.5 | **3.7M** |
| PIDNet-S | 0.881 | 0.820 | **24.2** | **41.3** | 7.6M |
| DeepLabV3+ | 0.877 | 0.801 | 52.9 | 18.9 | 22.4M |
| HRNet-W18-Small-v2 | 0.863 | 0.763 | 96.6 | 10.3 | 5.6M |

**Key trade-off:** U-Net achieved the highest wire segmentation accuracy (Dice 0.889, Boundary F1 0.847), while YOLO26s-Sem achieved nearly the same Wire Dice (0.886) at about 24.5 ms native latency / 40.8 FPS. PIDNet-S was similarly fast, but with lower wire accuracy.

## Models represented

- U-Net
- U-Net++
- DeepLabV3+
- HRNet-W18-Small-v2
- SegFormer-B0
- PIDNet-S
- YOLO26s-Sem

## Result files

- [`final_accuracy_runtime_table.csv`](results/final_accuracy_runtime_table.csv) — compact final accuracy/runtime comparison
- [`final_test_ranking.csv`](results/final_test_ranking.csv) — pixel and topology-aware final-test metrics
- [`runtime_summary_all_sizes.csv`](results/runtime_summary_all_sizes.csv) — crop and native-resolution runtime measurements

## Repository layout

```text
src/benchmark/   dataset, architecture, loss, pixel/topology and geometry metrics
src/             reference training entry point
configs/         sanitized example configuration
results/         final benchmark tables
```

## Reproducibility notes

The original research workspace contained datasets, checkpoints, server-specific paths and intermediate experiment artifacts. Those are intentionally excluded from this public-release repository. The included YAML is a sanitized template: users must provide their own images, masks and split lists.

The benchmark expects a three-class semantic setup: **background (0), wire (1), mover (2)**. For the project's three-mask annotation format, wire pixels take priority over mover pixels in overlap regions.

## Related system

This benchmark supports the perception module of the final **Version 3** wire-harness manipulation system. The system-level portfolio showcase is available at [`wire-harness-ai-motion-planning`](https://github.com/Alirezamirbagheri/wire-harness-ai-motion-planning).

## License

No license is included yet. A license should only be selected after confirming that all included source files can be redistributed publicly.
