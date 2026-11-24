"""单张图片猪检测推理脚本。

用法示例：
    python predict.py --image third_party/PigBench/detection/data/example_images/demo_image1.jpg \
        --weights yolov8_s --conf 0.4

输出：
    outputs/<图片名>_pred.jpg    带检测框和置信度的可视化图
    outputs/<图片名>_pred.json   每个框的坐标、置信度、计数、单张耗时
    stdout                       猪只计数、权重路径、设备、耗时

置信度默认值说明：0.4 取自官方 ``detection/tools/inference/inference_demo.ipynb``
中 ``draw_bboxes(..., score_thresh=0.4)`` 的推理/可视化默认值；
官方评估 ``test.py`` 使用的 ``score_thr=0.001`` 是 mAP 计算阈值，不用于本脚本的计数。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 允许从任意工作目录直接运行 python predict.py
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pigbench_infer import PigDetector, draw_detections, set_seed  # noqa: E402
from pigbench_infer.registry import DEFAULT_MODEL, MODELS  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PigBench 猪检测单张推理（仅推理，不训练）")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument(
        "--weights",
        default=DEFAULT_MODEL,
        help=f"模型 key（{', '.join(MODELS)}）或 .pth 路径，默认 {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="置信度阈值，默认 0.4（官方 inference_demo 的可视化默认值）",
    )
    parser.add_argument("--device", default="cpu", help="推理设备，默认 cpu（本机无 CUDA）")
    parser.add_argument("--output-dir", default="outputs", help="输出目录，默认 outputs/")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(42)

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        print(f"[predict] 错误：图片不存在 {image_path}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[predict] 数据量: 1 张图片")
    detector = PigDetector(weights=args.weights, device=args.device).load()

    t0 = time.perf_counter()
    result = detector.predict(image_path, conf=args.conf)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    stem = image_path.stem
    vis_path = output_dir / f"{stem}_vis.jpg"
    json_path = output_dir / f"{stem}.json"

    draw_detections(image_path, result["detections"], save_path=vis_path)

    payload = {
        "image": str(image_path),
        "model": result["model"],
        "checkpoint": result["checkpoint"],
        "device": result["device"],
        "conf_threshold": args.conf,
        "count": result["count"],
        "elapsed_ms": result["elapsed_ms"],
        "wall_ms": round(wall_ms, 2),
        "detections": result["detections"],
        "visualization": str(vis_path),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[predict] 权重: {result['checkpoint']}")
    print(f"[predict] 设备: {result['device']} | 输入尺寸: 见官方 config test_pipeline")
    print(f"[predict] 猪只计数: {result['count']}（conf >= {args.conf}）")
    print(f"[predict] 单张耗时: {result['elapsed_ms']:.2f} ms（模型内部计时）")
    print(f"[predict] 可视化: {vis_path}")
    print(f"[predict] JSON   : {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
