"""批量猪检测推理脚本（针对 inputs/ 文件夹）。

用法示例：
    python batch_predict.py --images-dir inputs --weights yolov8_s --conf 0.4

每张图输出：
    outputs/<图片名>_vis.jpg   带检测框和置信度的可视化图
    outputs/<图片名>.json      每个框的 xyxy 坐标、置信度、计数、耗时
汇总输出：
    results/batch_results.csv  列：文件名、检出数量、最高置信度、平均置信度、耗时

说明：
- 只做推理，不含训练入口；
- 不做精度评估（inputs/ 没有标注真值），因此不会输出 mAP/准确率之类数字；
- 置信度默认 0.4，取自官方 inference_demo.ipynb 的可视化默认阈值。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pigbench_infer import PigDetector, draw_detections, set_seed  # noqa: E402
from pigbench_infer.registry import DEFAULT_MODEL, MODELS  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PigBench YOLOv8 猪检测批量推理（仅推理）")
    parser.add_argument("--images-dir", default="inputs", help="待推理图片目录，默认 inputs/")
    parser.add_argument("--weights", default=DEFAULT_MODEL, help=f"模型 key 或 .pth 路径，默认 {DEFAULT_MODEL}")
    parser.add_argument("--conf", type=float, default=0.4, help="置信度阈值，默认 0.4")
    parser.add_argument("--device", default="cpu", help="推理设备，默认 cpu")
    parser.add_argument("--output-dir", default="outputs", help="可视化图与 JSON 输出目录，默认 outputs/")
    parser.add_argument("--csv", default="results/batch_results.csv", help="汇总 CSV 路径")
    return parser.parse_args(argv)


def list_images(images_dir: Path) -> list[Path]:
    """按文件名排序，保证每次运行顺序一致。"""
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(42)

    images_dir = Path(args.images_dir).expanduser()
    if not images_dir.is_absolute():
        images_dir = (PROJECT_ROOT / images_dir).resolve()
    if not images_dir.is_dir():
        print(f"[batch] 错误：图片目录不存在 {images_dir}", file=sys.stderr)
        return 2
    images = list_images(images_dir)
    if not images:
        print(f"[batch] 错误：{images_dir} 里没有图片（支持 {sorted(IMAGE_EXTS)}）", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = (PROJECT_ROOT / csv_path).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[batch] 数据量: {len(images)} 张图片 | 目录: {images_dir}")
    detector = PigDetector(weights=args.weights, device=args.device).load()

    rows: list[dict[str, object]] = []
    latencies: list[float] = []
    t_start = time.perf_counter()
    for idx, img in enumerate(images, 1):
        result = detector.predict(img, conf=args.conf)
        scores = [d["score"] for d in result["detections"]]
        stem = img.stem

        vis_path = output_dir / f"{stem}_vis.jpg"
        json_path = output_dir / f"{stem}.json"
        draw_detections(img, result["detections"], save_path=vis_path)
        payload = {
            "image": str(img),
            "model": result["model"],
            "checkpoint": result["checkpoint"],
            "device": result["device"],
            "conf_threshold": args.conf,
            "count": result["count"],
            "elapsed_ms": result["elapsed_ms"],
            "detections": result["detections"],
            "visualization": str(vis_path),
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        rows.append(
            {
                "filename": img.name,
                "count": result["count"],
                "max_score": round(max(scores), 4) if scores else "",
                "mean_score": round(statistics.fmean(scores), 4) if scores else "",
                "elapsed_ms": result["elapsed_ms"],
            }
        )
        latencies.append(result["elapsed_ms"])
        print(f"[batch] [{idx}/{len(images)}] {img.name}: 检出 {result['count']} 头, "
              f"{result['elapsed_ms']:.1f} ms -> {vis_path.name}, {json_path.name}")
    total_wall = time.perf_counter() - t_start

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "count", "max_score", "mean_score", "elapsed_ms"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[batch] 权重: {detector.spec['checkpoint']}")
    print(f"[batch] 设备: {args.device} | conf: {args.conf}")
    print(f"[batch] 平均单张: {statistics.fmean(latencies):.2f} ms | 最慢: {max(latencies):.2f} ms "
          f"| 总耗时: {total_wall:.1f} s")
    print(f"[batch] CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
