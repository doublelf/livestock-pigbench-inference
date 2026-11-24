"""视频逐帧猪检测（只做逐帧检测，不做跟踪/不分配跨帧 ID）。

检测逻辑完全复用 ``pigbench_infer.PigDetector``（与 predict.py 同一权重、同一后处理、同一阈值），
本脚本只负责：读视频 → 逐帧调用检测 → 画框/计数 → 写回视频 + CSV + JSON。

用法示例：
    python video_predict.py --video inputs/pigtrack0001.mp4 --weights yolov8_s --conf 0.4 --stride 1

输出：
    outputs/<视频名>_det.mp4          带框与 count 角标的视频（mp4v，原分辨率/帧率）
    results/video_counts.csv          只含实际检测帧的 frame_idx/timestamp_sec/count
    results/video_summary.json        总帧数、检测帧数、stride、计数统计、耗时
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pigbench_infer import PigDetector, draw_boxes_on_image, draw_count_overlay, set_seed  # noqa: E402
from pigbench_infer.registry import DEFAULT_MODEL, MODELS  # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PigBench 视频逐帧猪检测（非跟踪）")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--weights", default=DEFAULT_MODEL, help=f"模型 key 或 .pth 路径，默认 {DEFAULT_MODEL}")
    parser.add_argument("--conf", type=float, default=0.4, help="置信度阈值，默认 0.4（与图片推理一致）")
    parser.add_argument("--stride", type=int, default=1, help="每 N 帧检测一次，默认 1（每帧都检测）")
    parser.add_argument("--device", default="cpu", help="推理设备，默认 cpu")
    parser.add_argument("--output-dir", default="outputs", help="视频输出目录，默认 outputs/")
    parser.add_argument("--counts-csv", default="results/video_counts.csv", help="逐帧计数 CSV")
    parser.add_argument("--summary-json", default="results/video_summary.json", help="汇总 JSON")
    return parser.parse_args(argv)


def resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(42)

    if args.stride < 1:
        print("[video] 错误：--stride 必须 >= 1", file=sys.stderr)
        return 2

    video_path = resolve(args.video)
    if not video_path.is_file():
        print(f"[video] 错误：视频不存在 {video_path}", file=sys.stderr)
        print(f"[video] 请把 mp4 放进 {PROJECT_ROOT / 'inputs'}", file=sys.stderr)
        return 2
    if video_path.suffix.lower() not in VIDEO_EXTS:
        print(f"[video] 错误：不支持的视频后缀 {video_path.suffix}", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[video] 错误：OpenCV 无法打开 {video_path}", file=sys.stderr)
        return 2

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0:
        print(f"[video] 警告：读不到有效 fps（{fps!r}），回退为 25")
        fps = 25.0
    if width <= 0 or height <= 0:
        print("[video] 错误：读不到视频分辨率", file=sys.stderr)
        cap.release()
        return 2

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_path.stem}_det.mp4"

    counts_path = resolve(args.counts_csv)
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = resolve(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[video] 输入: {video_path}")
    print(f"[video] 规格: {width}x{height} | fps={fps:.3f} | 元数据帧数={reported_frames} | "
          f"时长约 {reported_frames / fps:.2f} s")
    print(f"[video] stride={args.stride} | conf={args.conf} | device={args.device}")
    print(f"[video] 输出视频: {out_path}")

    detector = PigDetector(weights=args.weights, device=args.device).load()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"[video] 错误：VideoWriter 打开失败 {out_path}", file=sys.stderr)
        cap.release()
        return 2

    rows: list[dict[str, float | int]] = []
    counts: list[int] = []
    infer_ms: list[float] = []
    frame_idx = 0
    detected_frames = 0
    last_count = 0
    t_start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detected_this_frame = (frame_idx % args.stride == 0)
            if detected_this_frame:
                result = detector.predict_frame(frame, conf=args.conf, frame_idx=frame_idx)
                last_count = result["count"]
                draw_boxes_on_image(frame, result["detections"])
                draw_count_overlay(frame, result["count"])
                rows.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_sec": round(frame_idx / fps, 2),
                        "count": result["count"],
                    }
                )
                counts.append(result["count"])
                infer_ms.append(result["elapsed_ms"])
                detected_frames += 1

            # 未检测的帧保持原始画面，不复制上一帧的框
            writer.write(frame)
            processed = frame_idx + 1
            if processed % 100 == 0:
                elapsed = time.perf_counter() - t_start
                cur = f"{last_count} 头" if detected_this_frame else "跳过（stride）"
                print(f"[video] 已处理 {processed} 帧 | 当前帧检出 {cur} | 已用 {elapsed:.1f} s")
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    total_wall = time.perf_counter() - t_start
    total_frames = frame_idx

    # 校验输出视频
    out_cap = cv2.VideoCapture(str(out_path))
    out_ok = out_cap.isOpened()
    out_frames = int(out_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if out_ok else -1
    out_fps = out_cap.get(cv2.CAP_PROP_FPS) if out_ok else float("nan")
    out_w = int(out_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if out_ok else -1
    out_h = int(out_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if out_ok else -1
    can_read, _ = out_cap.read() if out_ok else (False, None)
    out_cap.release()

    with counts_path.open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=["frame_idx", "timestamp_sec", "count"])
        wr.writeheader()
        wr.writerows(rows)

    summary = {
        "video": str(video_path),
        "output_video": str(out_path),
        "model": detector.spec["name"],
        "checkpoint": detector.spec["checkpoint"],
        "device": args.device,
        "conf_threshold": args.conf,
        "fps": round(float(fps), 3),
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "reported_frames_metadata": reported_frames,
        "detected_frames": detected_frames,
        "stride": args.stride,
        "avg_count": round(statistics.fmean(counts), 4) if counts else 0.0,
        "max_count": max(counts) if counts else 0,
        "min_count": min(counts) if counts else 0,
        "total_elapsed_sec": round(total_wall, 3),
        "avg_ms_per_frame": round(total_wall * 1000.0 / detected_frames, 2) if detected_frames else 0.0,
        "avg_inference_ms": round(statistics.fmean(infer_ms), 2) if infer_ms else 0.0,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output_check": {
            "opened": bool(out_ok),
            "frames": out_frames,
            "fps": round(float(out_fps), 3) if math.isfinite(out_fps) else None,
            "width": out_w,
            "height": out_h,
            "first_frame_readable": bool(can_read),
        },
        "note": "逐帧独立检测结果，帧间计数会有小幅跳动；没有跨帧 ID，不是多目标跟踪。",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[video] 完成：总帧数 {total_frames} | 检测帧 {detected_frames} | "
          f"平均计数 {summary['avg_count']:.2f}（min {summary['min_count']}, max {summary['max_count']}）")
    print(f"[video] 耗时 {total_wall:.2f} s | 平均每检测帧 "
          f"{summary['avg_ms_per_frame']:.2f} ms（模型推理均值 {summary['avg_inference_ms']:.2f} ms）")
    print(f"[video] 输出校验：opened={out_ok} frames={out_frames}（预期 {total_frames}） "
          f"fps={out_fps:.3f} size={out_w}x{out_h} first_frame_readable={can_read}")
    print(f"[video] 计数 CSV: {counts_path}")
    print(f"[video] 汇总 JSON: {summary_path}")

    if not out_ok or not can_read or out_frames != total_frames:
        print("[video] 警告：输出视频校验未完全通过，请查看 video_report.md", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
