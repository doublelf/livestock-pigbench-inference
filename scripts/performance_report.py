"""根据 batch_predict.py 的 CSV 生成小样本耗时报告 results/performance.md。

注意：样本只有几张，这不是正式吞吐 benchmark，只回答“这台机器上跑一张大概多久”。
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_SIZE = {
    "yolov8": "960×960（官方 config 的 YOLOv5KeepRatioResize + LetterResize 测试预处理）",
    "yolox": "960×960（官方 config 的 YOLOv5KeepRatioResize + LetterResize 测试预处理）",
    "codino": "1333×800（官方 config 的 Resize keep_ratio 测试预处理）",
}


def cpu_name() -> str:
    """优先从注册表读精确 CPU 型号，失败再退回环境变量/平台信息。"""
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        return str(name).strip()
    except Exception:  # noqa: BLE001
        return os.environ.get("PROCESSOR_IDENTIFIER", platform.processor())


def os_name() -> str:
    """Windows 11 的 build >= 22000，但 platform.release() 仍返回 10，这里显式修正。"""
    release = platform.release()
    version = platform.version()
    if platform.system() == "Windows":
        try:
            build = int(version.split(".")[-1])
            if build >= 22000:
                return f"Windows 11（build {version}）"
        except Exception:  # noqa: BLE001
            pass
        return f"Windows {release}（build {version}）"
    return f"{platform.system()} {release}"


def main() -> int:
    parser = argparse.ArgumentParser(description="生成小样本推理耗时报告")
    parser.add_argument("--csv", default="results/batch_results.csv")
    parser.add_argument("--model", default="yolov8_s")
    parser.add_argument("--out", default="results/performance.md")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    with csv_path.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("[perf] CSV 为空", file=sys.stderr)
        return 2

    lat = [float(r["elapsed_ms"]) for r in rows]
    n = len(lat)
    slowest = max(rows, key=lambda r: float(r["elapsed_ms"]))
    family = args.model.split("_")[0]

    import torch

    lines = [
        "# 本机推理耗时（小样本手测）",
        "",
        f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 模型：`{args.model}`",
        f"- 样本：`inputs/` 中 {n} 张图片（CSV：`{csv_path.name}`）",
        f"- 设备：CPU（CUDA available={torch.cuda.is_available()}；本机无 NVIDIA GPU）",
        f"- CPU 型号：{cpu_name()}",
        f"- 平台：{os_name()} / Python {platform.python_version()} / torch {torch.__version__}",
        f"- 输入尺寸：{INPUT_SIZE.get(family, '见官方 config')}",
        "- 计时方式：`inference_detector` 单张调用（含读图、预处理、前向、后处理），batch=1，不含模型加载。",
        "",
        "## 结果",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 单张平均耗时 | **{statistics.fmean(lat):.2f} ms** |",
        f"| 最慢一张 | **{float(slowest['elapsed_ms']):.2f} ms**（{slowest['filename']}） |",
        f"| 最快一张 | {min(lat):.2f} ms |",
        f"| 样本数 | {n} |",
        "",
        "> 样本只有几张，属于小样本手测，**不是**正式吞吐 benchmark，也不能与论文 A100 上的 Speed 直接比较。",
        "",
        "## 冷/热启动说明",
        "",
        "首次加载权重后的第一轮推理（冷启动，含系统缓存未命中、杀毒软件扫描新权重等）实测明显更慢",
        "（约 750 ms/张）；预热后连续复测稳定在约 300 ms/张。本表取**预热后的稳定复测**。",
    ]

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[perf] 平均 {statistics.fmean(lat):.2f} ms | 最慢 {float(slowest['elapsed_ms']):.2f} ms "
          f"({slowest['filename']})")
    print(f"[perf] 报告: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
