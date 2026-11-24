# 本机推理耗时（小样本手测）

- 生成时间：2026-11-24T16:15:34+08:00
- 模型：`yolov8_s`
- 样本：`inputs/` 中 5 张图片（CSV：`batch_results.csv`）
- 设备：CPU（CUDA available=False；本机无 NVIDIA GPU）
- CPU 型号：13th Gen Intel(R) Core(TM) i5-13500H
- 平台：Windows 11（build 10.0.26200） / Python 3.10.11 / torch 2.0.0+cpu
- 输入尺寸：960×960（官方 config 的 YOLOv5KeepRatioResize + LetterResize 测试预处理）
- 计时方式：`inference_detector` 单张调用（含读图、预处理、前向、后处理），batch=1，不含模型加载。

## 结果

| 指标 | 数值 |
| --- | --- |
| 单张平均耗时 | **300.44 ms** |
| 最慢一张 | **316.26 ms**（pig_1.png） |
| 最快一张 | 287.04 ms |
| 样本数 | 5 |

> 样本只有几张，属于小样本手测，**不是**正式吞吐 benchmark，也不能与论文 A100 上的 Speed 直接比较。

## 冷/热启动说明

首次加载权重后的第一轮推理（冷启动，含系统缓存未命中、杀毒软件扫描新权重等）实测明显更慢
（约 750 ms/张）；预热后连续复测稳定在约 300 ms/张。本表取**预热后的稳定复测**。
