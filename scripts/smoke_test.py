"""环境冒烟测试：不加载任何预训练权重，只用官方 config 构建 YOLOv8-S 并跑一张图。

目的：在权重到位前，验证 mmdet/mm yolo 在本机（Windows + CPU）能正常 import、构建模型、
执行官方 test_pipeline 与后处理。随机初始化权重，输出没有语义意义，不能当作推理结果。

用法：python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

from pigbench_infer.registry import DETECTION_DIR  # noqa: E402


def main() -> int:
    from mmdet.apis import inference_detector, init_detector
    from mmdet.utils import register_all_modules

    cfg = DETECTION_DIR / "configs" / "yolov8" / "yolov8_s.py"
    img = DETECTION_DIR / "data" / "example_images" / "demo_image1.jpg"
    if not cfg.is_file() or not img.is_file():
        print(f"[smoke] 缺少官方文件：{cfg} / {img}", file=sys.stderr)
        return 2

    print(f"[smoke] config: {cfg}")
    print(f"[smoke] image : {img}")
    print("[smoke] device: cpu | checkpoint: None（随机初始化，仅验证流程）")

    register_all_modules(init_default_scope=False)
    t0 = time.perf_counter()
    model = init_detector(str(cfg), None, device="cpu")
    model.eval()
    print(f"[smoke] 模型构建成功，参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M，"
          f"耗时 {time.perf_counter() - t0:.2f} s")

    t1 = time.perf_counter()
    result = inference_detector(model, str(img))
    inst = result.pred_instances
    print(f"[smoke] 推理完成：{len(inst.scores)} 个候选框（随机权重，数量无意义），"
          f"耗时 {time.perf_counter() - t1:.2f} s")

    # 用 OpenCV 读一下图片，确认尺寸与可视化依赖正常
    image = cv2.imread(str(img))
    print(f"[smoke] 图片尺寸: {image.shape[1]}x{image.shape[0]}")
    print("[smoke] PASS：环境、config、推理 API 均可用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
