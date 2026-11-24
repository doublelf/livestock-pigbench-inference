"""校验本次唯一权重 yolov8_s.pth（只做本地检查，不下载）。

检查项：
1. 文件是否存在、实际大小、SHA256；
2. torch.load 是否可读、state_dict 键数量；
3. 官方 init_detector 是否能加载、参数量、分类头类别数（应为 1，排除 COCO 权重）。

输出 results/files_check.md。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pigbench_infer.registry import DETECTION_DIR, resolve_model  # noqa: E402

MB = 1_000_000


def sha256sum(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 yolov8_s.pth")
    parser.add_argument("--weights", default="yolov8_s", help="注册表 key 或 .pth 路径")
    parser.add_argument("--out", default="results/files_check.md")
    args = parser.parse_args()

    spec = resolve_model(args.weights)
    ckpt = Path(spec["checkpoint"])
    cfg = Path(spec["config"])

    import torch
    import mmcv
    import mmdet
    import mmyolo

    lines = [
        "# 权重文件校验（files_check）",
        "",
        f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 模型 key：`{spec['name']}`",
        f"- 官方 config：`{cfg}`",
        f"- 权重路径：`{ckpt}`",
        f"- 环境：Python {sys.version.split()[0]} / torch {torch.__version__} / mmcv {mmcv.__version__} / "
        f"mmdet {mmdet.__version__} / mmyolo {mmyolo.__version__} / CUDA available={torch.cuda.is_available()}",
        "",
    ]

    if not ckpt.is_file():
        lines += [
            "## 结论：FAIL（文件不存在）",
            "",
            f"请从官方直达链接重新下载：",
            "",
            "https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/I6UYE9/PQWVVN",
            "",
            f"放置到：`{ckpt}`",
        ]
        out = PROJECT_ROOT / args.out
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[check] 缺失：{ckpt}")
        return 1

    size = ckpt.stat().st_size
    size_mb = size / MB
    lines += [
        "## 文件信息",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| 文件 | `{ckpt.name}` |",
        f"| 实际大小 | **{size_mb:.2f} MB**（{size} 字节） |",
        f"| 官方口径 | 官方 README 仅给 YOLOv8 pig weights 整组 573 MB（S/M/L/X 四个文件合计），单文件未单列 |",
        f"| SHA256 | `{sha256sum(ckpt)}` |",
        "",
    ]

    # 合理性检查：YOLOv8-S 约 11.1M 参数，fp32 state_dict 约 44 MB；不把区间当官方数字
    size_ok = 20 <= size_mb <= 200
    lines.append(
        f"- 大小合理性：{'通过' if size_ok else '异常'}（YOLOv8-S 约 11.1M 参数，fp32 权重预计几十 MB；"
        f"当前 {size_mb:.2f} MB）。"
    )

    # torch.load
    load_ok = True
    key_count = -1
    ckpt_classes = None
    ckpt_meta = {}
    try:
        obj = torch.load(str(ckpt), map_location="cpu")
        sd = obj.get("state_dict", obj)
        key_count = len(sd)
        ckpt_meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
        ckpt_classes = (ckpt_meta.get("dataset_meta") or {}).get("classes")
        lines.append(f"- `torch.load`：成功，state_dict 共 {key_count} 个键。")
        if ckpt_classes:
            lines.append(f"- checkpoint 内记录的类别：`{tuple(ckpt_classes)}`")
        if ckpt_meta:
            lines.append(
                f"- checkpoint 元信息：epoch=`{ckpt_meta.get('epoch')}`，"
                f"experiment=`{ckpt_meta.get('experiment_name')}`"
            )
    except Exception as exc:  # noqa: BLE001
        load_ok = False
        lines.append(f"- `torch.load`：**失败**，{type(exc).__name__}: {exc}")

    # 官方 init_detector
    init_ok = False
    n_params = 0
    cls_out = None
    err = ""
    try:
        from mmdet.apis import init_detector
        from mmdet.utils import register_all_modules

        register_all_modules(init_default_scope=False)
        model = init_detector(str(cfg), str(ckpt), device="cpu")
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        try:
            head = model.bbox_head
            if hasattr(head, "head_module") and hasattr(head.head_module, "cls_preds"):
                # cls_preds[-1] 通常是 nn.Sequential，取其中最后一个 Conv2d 的输出通道
                for module in head.head_module.cls_preds[-1].modules():
                    if isinstance(module, torch.nn.Conv2d):
                        cls_out = module.out_channels
        except Exception:  # noqa: BLE001
            pass
        del model
        init_ok = True
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"

    lines += [
        "",
        "## 加载检查（官方 init_detector，device=cpu）",
        "",
        f"- 加载结果：**{'PASS' if init_ok else 'FAIL'}**" + (f"（{err}）" if err else ""),
        f"- 参数量：{n_params / 1e6:.2f} M" if init_ok else "- 参数量：-",
        f"- 分类头 out_channels：{cls_out}（猪检测应为 1；若是 80 说明拿成了 COCO 权重）"
        if cls_out is not None
        else "- 分类头 out_channels：未取到",
        "",
    ]

    ok = load_ok and init_ok and size_ok and (cls_out in (None, 1))
    lines += [
        "## 结论",
        "",
        "**PASS**：文件存在、可被 PyTorch 读取、可被官方 `init_detector` 正常加载"
        + ("，分类头类别数为 1（是 pig 权重而非 COCO 权重）。" if cls_out == 1 else "。")
        if ok
        else "**FAIL**：见上表，请不要继续推理，按直达链接重下或检查是否放错目录。",
    ]

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[check] size={size_mb:.2f} MB | init={init_ok} | params={n_params} | cls_out={cls_out}")
    print(f"[check] 报告: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
