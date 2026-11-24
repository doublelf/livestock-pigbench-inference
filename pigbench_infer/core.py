"""官方 mmdet/mm yolo 模型的推理封装。

关键点：
- 模型构建与推理全部走官方 API：``init_detector`` + ``inference_detector``；
- 预处理/后处理完全由官方 config 的 test_pipeline 与 test_cfg 决定，本项目不重写；
- 不包含训练入口，不调用任何 ``train*`` 脚本。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .registry import DETECTION_DIR, resolve_model

# 固定随机种子（抽样、可视化选图统一使用）
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """固定 python/numpy/torch 的随机种子，保证抽样可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # 仅做推理时 torch 不是必需的导入前置；导入失败不影响抽样复现
        pass


@dataclass
class Detection:
    """单个检测框。坐标为原图像素坐标 [x1, y1, x2, y2]。"""

    bbox: list[float]
    score: float
    label: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"bbox": [round(v, 2) for v in self.bbox], "score": round(self.score, 4)}


class PigDetector:
    """加载官方 pig 权重，对单张/多张图片做前向推理。"""

    def __init__(self, weights: str = "yolov8_s", device: str = "cpu"):
        self.spec = resolve_model(weights)
        self.device = device
        self._model = None

    # ------------------------------------------------------------------ #
    def load(self) -> "PigDetector":
        """构建模型并加载官方预训练权重（只加载，不训练）。"""
        # 延迟导入：保证没有装依赖时，本包的纯工具函数仍可被 import
        import mmdet  # noqa: F401
        import mmyolo  # noqa: F401  # 注册 YOLO 系列模块
        from mmdet.apis import init_detector
        from mmdet.utils import register_all_modules

        cfg = self.spec["config"]
        ckpt = self.spec["checkpoint"]
        if not Path(cfg).is_file():
            raise FileNotFoundError(f"官方 config 不存在：{cfg}")
        if not Path(ckpt).is_file():
            raise FileNotFoundError(f"官方权重不存在：{ckpt}")

        print(f"[PigDetector] config     : {cfg}")
        print(f"[PigDetector] checkpoint : {ckpt}")
        print(f"[PigDetector] device     : {self.device}")

        register_all_modules(init_default_scope=False)
        t0 = time.perf_counter()
        self._model = init_detector(cfg, ckpt, device=self.device)
        self._model.eval()
        print(f"[PigDetector] 权重加载耗时: {time.perf_counter() - t0:.2f} s")
        return self

    # ------------------------------------------------------------------ #
    def _build_result(
        self, result: Any, elapsed_ms: float, conf: float, image_label: str
    ) -> dict[str, Any]:
        """把官方 ``inference_detector`` 的返回值整理成本项目统一结构。"""
        inst = result.pred_instances
        bboxes = inst.bboxes.detach().cpu().numpy()
        scores = inst.scores.detach().cpu().numpy()
        labels = inst.labels.detach().cpu().numpy() if hasattr(inst, "labels") else np.zeros(len(scores))

        keep = scores >= conf
        dets = [
            Detection(bbox=bboxes[i].tolist(), score=float(scores[i]), label=int(labels[i]))
            for i in np.where(keep)[0]
        ]
        # 按置信度从高到低排序，方便报告和画图
        dets.sort(key=lambda d: d.score, reverse=True)

        return {
            "image": image_label,
            "model": self.spec["name"],
            "config": self.spec["config"],
            "checkpoint": self.spec["checkpoint"],
            "device": self.device,
            "conf_threshold": conf,
            "count": len(dets),
            "elapsed_ms": round(elapsed_ms, 2),
            "detections": [d.as_dict() for d in dets],
        }

    # ------------------------------------------------------------------ #
    def predict(self, image_path: str | Path, conf: float = 0.4) -> dict[str, Any]:
        """对单张图片推理，返回框、置信度、计数与耗时（毫秒）。"""
        if self._model is None:
            self.load()

        from mmdet.apis import inference_detector

        image_path = Path(image_path)
        t0 = time.perf_counter()
        result = inference_detector(self._model, str(image_path))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._build_result(result, elapsed_ms, conf, str(image_path))

    # ------------------------------------------------------------------ #
    def predict_frame(
        self, frame_bgr: np.ndarray, conf: float = 0.4, frame_idx: int | None = None
    ) -> dict[str, Any]:
        """对内存中的 BGR 帧推理（视频逐帧用），与 ``predict`` 共用同一套后处理。

        注意：这里传的是 OpenCV 读出的 BGR 帧，和官方 ``LoadImageFromFile`` 的通道顺序一致；
        预处理仍由官方 config 的 test_pipeline 完成，封装层不改。
        """
        if self._model is None:
            self.load()

        from mmdet.apis import inference_detector

        t0 = time.perf_counter()
        result = inference_detector(self._model, frame_bgr)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        label = f"frame:{frame_idx}" if frame_idx is not None else "<in-memory>"
        return self._build_result(result, elapsed_ms, conf, label)

    # ------------------------------------------------------------------ #
    def warmup(self, image_path: str | Path, conf: float = 0.4, runs: int = 2) -> None:
        """预热：排除首次推理的初始化开销（计时统计前调用）。"""
        for _ in range(runs):
            self.predict(image_path, conf=conf)


def draw_boxes_on_image(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    color: tuple[int, int, int] = (0, 0, 255),
    label_prefix: str = "",
    show_scores: bool = True,
    linewidth: int = 2,
) -> np.ndarray:
    """在给定的 BGR 图像上画检测框（原地修改并返回），视频逐帧复用。

    说明：官方 notebook 的可视化基于 matplotlib；这里改用 OpenCV 只是为了批量/视频出图更快，
    框坐标与置信度完全一致。
    """
    h, w = image.shape[:2]
    thick = max(1, int(linewidth))
    font_scale = max(0.4, min(w, h) / 1200.0)

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thick)
        if show_scores:
            text = f"{label_prefix}{det['score']:.2f}"
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            ty = max(y1, th + 2)
            cv2.rectangle(image, (x1, ty - th - 2), (x1 + tw + 4, ty + base), color, -1)
            cv2.putText(
                image, text, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 1, cv2.LINE_AA,
            )
    return image


def draw_count_overlay(image: np.ndarray, count: int) -> np.ndarray:
    """在画面左上角叠加当前帧猪只计数（视频用）。"""
    text = f"count: {count}"
    font_scale = max(0.8, min(image.shape[1], image.shape[0]) / 900.0)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    cv2.rectangle(image, (8, 8), (8 + tw + 12, 8 + th + base + 10), (0, 0, 0), -1)
    cv2.putText(
        image, text, (14, 8 + th + 4), cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, (0, 255, 255), 2, cv2.LINE_AA,
    )
    return image


def draw_detections(
    image_path: str | Path,
    detections: list[dict[str, Any]],
    save_path: str | Path | None = None,
    color: tuple[int, int, int] = (0, 0, 255),
    label_prefix: str = "",
    show_scores: bool = True,
    linewidth: int = 2,
) -> np.ndarray:
    """读取图片 → 画框 → 可选保存，返回绘制后的图像。"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}")
    image = draw_boxes_on_image(
        image, detections, color=color, label_prefix=label_prefix,
        show_scores=show_scores, linewidth=linewidth,
    )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), image)
    return image


def load_gt_boxes(coco_json: str | Path) -> dict[str, list[list[float]]]:
    """读取 COCO 标注，返回 {文件名: [[x1,y1,x2,y2], ...]}。

    只用标准库 json，避免依赖 pycocotools。
    """
    import json

    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = {im["id"]: im["file_name"] for im in data["images"]}
    anns: dict[str, list[list[float]]] = {name: [] for name in images.values()}
    for ann in data.get("annotations", []):
        name = images.get(ann["image_id"])
        if name is None:
            continue
        x, y, w, h = ann["bbox"]
        anns[name].append([float(x), float(y), float(x + w), float(y + h)])
    return anns


def resolve_image_dir() -> Path:
    """PigDetect test 图片目录（restructure 之后所有图片都在 images/ 下）。"""
    return DETECTION_DIR / "data" / "PigDetect" / "images"
