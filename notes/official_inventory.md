# PigBench 官方资料清单（official_inventory）

> 本期范围：**只做 YOLOv8-S 单权重本地推理部署**。不训练、不下载数据集、不算 mAP、不做跟踪。
> 本文件只记录官方仓库/论文中的事实，文件名与命令以官方文档实际写法为准。
> 采集时间：2026-11-24；代码来自 `git clone https://github.com/jonaden94/PigBench.git`（仅代码）。

## 1. 官方来源

| 项目 | 地址 |
| --- | --- |
| 代码仓库 | https://github.com/jonaden94/PigBench |
| 本地克隆路径 | `third_party/PigBench` |
| 本地 commit | `7967036065fa5eceffb67c74a8d631711d27e827`（2026-11-11） |
| 检测指南 | `third_party/PigBench/detection/README.md` |
| 权重 DOI | https://doi.org/10.25625/I6UYE9 （跳转到 data.goettingen-research-online.de） |
| 论文 | arXiv:2507.16639；Computers and Electronics in Agriculture 241 (2026) 111264 |

## 2. 本期唯一使用的模型与权重

| 项目 | 官方写法 |
| --- | --- |
| 模型 | YOLOv8-S（官方 README 中 YOLOv8 pig weights 的最小一档） |
| 官方 config | `configs/yolov8/yolov8_s.py`（相对 `detection/`） |
| 官方权重路径 | `data/pretrained_weights/yolov8_pigs/yolov8_s.pth`（相对 `detection/`） |
| 权重文件 ID | `PQWVVN` |
| 官方下载命令（**本项目不执行**） | `python tools/download/download.py --name yolov8_weights --root data/pretrained_weights/yolov8_pigs` |
| 官方标注大小 | README 写“YOLOv8 pig weights 573 MB”，这是 S/M/L/X **四个文件合计**；单文件大小官方未单列 |
| 直达链接 | https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/I6UYE9/PQWVVN |

> 注意：`yolov8_pretrained_weights`（COCO 初始化权重，文件名含 `_coco_`）是训练用的，**本期不要**。
> 本期也不下 YOLOX（约 718 MB）与 Co-DINO（约 900 MB）pig weights。
> Hugging Face 镜像只有数据集，没有权重，不要去 HF 找权重。

## 3. 官方 config 关键信息（`configs/yolov8/yolov8_s.py`）

- 模型类：`YOLODetector`；backbone：`YOLOv8CSPDarknet`（arch=P5, deepen=0.33, widen=0.5）；neck：`YOLOv8PAFPN`。
- 类别：`metainfo = {'classes': ('pig',), 'palette': [(220, 20, 60)]}`，`num_classes = 1`。
- 测试预处理（`test_pipeline`）：
  `LoadImageFromFile` → `YOLOv5KeepRatioResize(scale=(960, 960))` → `LetterResize(scale=(960,960), allow_scale_up=False, pad_val=114)` → `LoadAnnotations` → `PackDetInputs`。
- 测试后处理（`model.test_cfg`）：
  `score_thr=0.001`、`nms=dict(type='nms', iou_threshold=0.75)`、`max_per_img=300`、`multi_label=False`。
- 这些值是官方评估/推理的共同底层配置；本项目不在封装层改写它们。

## 4. 官方推理入口与写法

官方 demo：`detection/tools/inference/inference_demo.ipynb`，核心代码：

```python
from mmdet.utils import register_all_modules
from mmdet.apis import init_detector, inference_detector

register_all_modules(init_default_scope=False)
model = init_detector(config_path, checkpoint_path, device='cuda:0')  # 无 GPU 用 'cpu'
result = inference_detector(model, image)   # 也支持图片路径 / 图片列表
scores = result.pred_instances.scores.cpu().numpy()
bboxes = result.pred_instances.bboxes.cpu().numpy()
```

- 官方 notebook 在备注里明确给出 CPU 备选：`configs/yolov8/yolov8_s.py` +
  `data/pretrained_weights/yolov8_pigs/yolov8_s.pth` + `device='cpu'`，这正是本期的组合。
- 可视化：`tools/inference/visualization_utils.py::draw_bboxes(...)`，notebook 调用时
  `score_thresh=0.4, linewidth=2, show_scores=True, figsize=(15,15), score_fontsize=6`。
- **阈值口径**：官方推理/demo 的可视化默认阈值是 **0.4**；config 里的 `score_thr=0.001`
  是评估阶段为了算 mAP 的召回阈值。本期 `predict.py`/`batch_predict.py` 的 `--conf` 默认取 **0.4**，
  并在 README 中写明出处，不自己发明新阈值。

## 5. 官方环境版本（`_setup/setup.sh` + `requirements.txt`）

- Python 3.10
- PyTorch 2.0.0 / torchvision 0.15.0
- CUDA 11.8（官方只在 Linux + CUDA 11.8 测试）
- mmengine（`mim install mmengine`，未锁小版本）
- mmcv 2.0.0
- mmdet 3.3.0
- mmyolo 0.6.0
- numpy 1.26.4
- 检测推理还用不到 MOTIP/MOTRv2 的自定义 CUDA 算子，官方也说明那部分“不是必需”。

本机无 NVIDIA GPU（仅 Intel Iris Xe），实际使用 CPU 版 torch 2.0.0；这是官方未测试的组合，
必须如实记录差异。实测版本见 `results/environment.md`。

## 6. 许可

- 仓库根目录：MIT。
- `detection/` 子目录：**GNU GPL v3**（`detection/LICENSE` 已确认是 GPLv3 文本）；
  该目录又组合了 mmdetection（Apache-2.0）与 mmyolo（GPL 3.0）代码。
- `tracking/boxmot/`：AGPL v3；`tracking/motip/`、`tracking/motrv2/`：Apache-2.0（本期不用）。
- 本项目引用 `detection/` 的 config 与推理代码，因此按 **GPLv3** 处理；仅定位为本地研发部署验证，
  不默认商用。数据集本期不使用，不涉及数据许可。

## 7. 文档缺口 / 注意事项

1. 官方 README 只给权重整组大小（4 个文件 573 MB），**没有单文件大小**；单文件完整性以
   DOI 下载页显示大小 + 本地加载校验为准。
2. 官方 setup 只保证 Linux + CUDA 11.8；本机 Windows + CPU 属未测试组合，跑通后仍需在报告中注明。
3. 官方 demo notebook 已被标记为 non-functional（Colab runtime 问题），但其 API 用法仍是官方入口；
   本项目直接调用同一套 `init_detector` / `inference_detector`，不用 notebook。
4. 官方权重站点在部分网络下会连接重置；按要求由用户手动下载，本项目不代下。
5. 论文 Table 5 有 PigDetect test 的 AP 数字（YOLOv8-S AP=80.3 / AP0.5=97.8），但本期**不做数据集评估**，
   这些数字只作为背景记录，不得当作本机结果。
