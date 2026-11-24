# 权重文件校验（files_check）

- 生成时间：2026-11-24T16:57:02+08:00
- 模型 key：`yolov8_s`
- 官方 config：`D:\Desktop\chat\opencode\project\livestock-pigbench-inference\third_party\PigBench\detection\configs\yolov8\yolov8_s.py`
- 权重路径：`D:\Desktop\chat\opencode\project\livestock-pigbench-inference\third_party\PigBench\detection\data\pretrained_weights\yolov8_pigs\yolov8_s.pth`
- 环境：Python 3.10.11 / torch 2.0.0+cpu / mmcv 2.0.0 / mmdet 3.3.0 / mmyolo 0.6.0 / CUDA available=False

## 文件信息

| 项目 | 值 |
| --- | --- |
| 文件 | `yolov8_s.pth` |
| 实际大小 | **44.77 MB**（44772733 字节） |
| 官方口径 | 官方 README 仅给 YOLOv8 pig weights 整组 573 MB（S/M/L/X 四个文件合计），单文件未单列 |
| SHA256 | `eb951f2ff765cc8cfb4aaabe03a7a02dd45669d5dd53db98d3baebb4ad16b32d` |

- 大小合理性：通过（YOLOv8-S 约 11.1M 参数，fp32 权重预计几十 MB；当前 44.77 MB）。
- `torch.load`：成功，state_dict 共 354 个键。
- checkpoint 内记录的类别：`('pig',)`
- checkpoint 元信息：epoch=`52`，experiment=`yolov8_s_20240425_120132`

## 加载检查（官方 init_detector，device=cpu）

- 加载结果：**PASS**
- 参数量：11.14 M
- 分类头 out_channels：1（猪检测应为 1；若是 80 说明拿成了 COCO 权重）

## 结论

**PASS**：文件存在、可被 PyTorch 读取、可被官方 `init_detector` 正常加载，分类头类别数为 1（是 pig 权重而非 COCO 权重）。
