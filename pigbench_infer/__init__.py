"""PigBench 检测推理封装。

本包只负责“加载官方预训练权重 + 跑官方推理 API + 结果后处理/可视化”，
不包含任何训练逻辑，也不修改 third_party/PigBench 内的官方代码。
"""

from .registry import MODELS, DETECTION_DIR, PROJECT_ROOT, resolve_model
from .core import (
    PigDetector,
    draw_boxes_on_image,
    draw_count_overlay,
    draw_detections,
    set_seed,
    load_gt_boxes,
)

__all__ = [
    "MODELS",
    "DETECTION_DIR",
    "PROJECT_ROOT",
    "resolve_model",
    "PigDetector",
    "draw_boxes_on_image",
    "draw_count_overlay",
    "draw_detections",
    "set_seed",
    "load_gt_boxes",
]
