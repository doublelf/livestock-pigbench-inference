"""模型/权重路径注册表。

所有路径都以官方 ``detection/`` 目录为基准，与官方 README 的写法保持一致。
权重文件名严格来自官方 ``detection/tools/download/download.py``，不做任何猜测。
"""

from __future__ import annotations

from pathlib import Path

# 项目根目录（本文件位于 <root>/pigbench_infer/registry.py）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 官方仓库的 detection 目录
DETECTION_DIR = PROJECT_ROOT / "third_party" / "PigBench" / "detection"

# key -> (config 相对路径, 权重相对路径, 家族)
MODELS: dict[str, dict[str, str]] = {
    "yolov8_s": {
        "config": "configs/yolov8/yolov8_s.py",
        "weights": "data/pretrained_weights/yolov8_pigs/yolov8_s.pth",
        "family": "yolov8",
    },
    "yolov8_m": {
        "config": "configs/yolov8/yolov8_m.py",
        "weights": "data/pretrained_weights/yolov8_pigs/yolov8_m.pth",
        "family": "yolov8",
    },
    "yolov8_l": {
        "config": "configs/yolov8/yolov8_l.py",
        "weights": "data/pretrained_weights/yolov8_pigs/yolov8_l.pth",
        "family": "yolov8",
    },
    "yolov8_x": {
        "config": "configs/yolov8/yolov8_x.py",
        "weights": "data/pretrained_weights/yolov8_pigs/yolov8_x.pth",
        "family": "yolov8",
    },
    "yolox_s": {
        "config": "configs/yolox/yolox_s_rtmdet_hyp.py",
        "weights": "data/pretrained_weights/yolox_pigs/yolox_s.pth",
        "family": "yolox",
    },
    "yolox_m": {
        "config": "configs/yolox/yolox_m_rtmdet_hyp.py",
        "weights": "data/pretrained_weights/yolox_pigs/yolox_m.pth",
        "family": "yolox",
    },
    "yolox_l": {
        "config": "configs/yolox/yolox_l.py",
        "weights": "data/pretrained_weights/yolox_pigs/yolox_l.pth",
        "family": "yolox",
    },
    "yolox_x": {
        "config": "configs/yolox/yolox_x.py",
        "weights": "data/pretrained_weights/yolox_pigs/yolox_x.pth",
        "family": "yolox",
    },
    "codino": {
        "config": "configs/co_detr/co_dino_swin.py",
        "weights": "data/pretrained_weights/codino_pigs/codino_swin.pth",
        "family": "codino",
    },
}

# 默认模型：官方 inference_demo.ipynb 在 CPU 备注里推荐的 YOLOv8-S
DEFAULT_MODEL = "yolov8_s"


def resolve_model(weights: str) -> dict[str, str]:
    """把 ``--weights`` 参数解析成 config / checkpoint 的绝对路径。

    支持两种写法：
    1. 注册表 key（推荐），如 ``yolov8_s``；
    2. 直接给 ``.pth`` 路径，按文件名推断家族与 config（仅限官方命名）。
    """
    if weights in MODELS:
        info = MODELS[weights]
        return {
            "name": weights,
            "config": str((DETECTION_DIR / info["config"]).resolve()),
            "checkpoint": str((DETECTION_DIR / info["weights"]).resolve()),
            "family": info["family"],
        }

    ckpt = Path(weights).expanduser()
    if not ckpt.is_absolute():
        ckpt = (Path.cwd() / ckpt).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"找不到权重文件：{ckpt}；--weights 也可直接传注册表 key（如 yolov8_s）"
        )

    name = ckpt.stem.lower()
    if name.startswith("yolov8_"):
        size = name.split("_")[-1]
        cfg = f"configs/yolov8/yolov8_{size}.py"
    elif name.startswith("yolox_"):
        size = name.split("_")[-1]
        # 官方 eval.sh 中 s/m 用 rtmdet_hyp 配置，l/x 用普通配置
        cfg = (
            f"configs/yolox/yolox_{size}_rtmdet_hyp.py"
            if size in {"s", "m"}
            else f"configs/yolox/yolox_{size}.py"
        )
    elif name == "codino_swin":
        cfg = "configs/co_detr/co_dino_swin.py"
    else:
        raise ValueError(
            f"无法从文件名推断官方 config：{ckpt.name}；请改用注册表 key：{sorted(MODELS)}"
        )

    cfg_path = (DETECTION_DIR / cfg).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"推断出的 config 不存在：{cfg_path}")
    return {
        "name": ckpt.stem,
        "config": str(cfg_path),
        "checkpoint": str(ckpt),
        "family": name.split("_")[0],
    }
