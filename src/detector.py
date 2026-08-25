"""Detection + raw tracking via YOLO-World + ByteTrack (ultralytics)."""
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLOWorld

from config import Config


@dataclass
class Detection:
    raw_id: int          # ByteTrack id (-1 if untracked this frame)
    xyxy: tuple          # (x1, y1, x2, y2) float
    conf: float

    @property
    def center(self):
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def diameter(self):
        x1, y1, x2, y2 = self.xyxy
        return max(x2 - x1, y2 - y1)


def mean_lab(frame_bgr: np.ndarray, xyxy, inner_frac: float) -> np.ndarray | None:
    """Mean Lab color of the central ellipse of the bbox — the flip signature feature."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    mask = np.zeros((ch, cw), np.uint8)
    cv2.ellipse(mask, (cw // 2, ch // 2),
                (max(1, int(cw * inner_frac / 2)), max(1, int(ch * inner_frac / 2))),
                0, 0, 360, 255, -1)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    return np.array(cv2.mean(lab, mask=mask)[:3], dtype=np.float32)


def in_roi(xyxy, roi) -> bool:
    if roi is None:
        return True
    cx = (xyxy[0] + xyxy[2]) / 2.0
    cy = (xyxy[1] + xyxy[3]) / 2.0
    x1, y1, x2, y2 = roi
    return x1 <= cx <= x2 and y1 <= cy <= y2


class PattyDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = YOLOWorld(cfg.model_name)
        self.model.set_classes(list(cfg.classes))

    def stream(self, video_path: str, use_tracker: bool = True):
        """Yield (frame_idx, frame_bgr, list[Detection]) for every frame.

        With use_tracker=False the detector reports every box it sees. ByteTrack
        will only *start* a track from a confident detection, so on a griddle
        where most patties sit near the open-vocabulary model's noise floor it
        silently swallows the majority of them — fine when the tracker owns
        identity, fatal when something downstream does.
        """
        common = dict(conf=self.cfg.conf, iou=self.cfg.iou, imgsz=self.cfg.imgsz,
                      device=self.cfg.device, verbose=False, agnostic_nms=True)
        if use_tracker:
            results = self.model.track(source=video_path, stream=True, persist=True,
                                       tracker=self.cfg.tracker, **common)
        else:
            results = self.model.predict(source=video_path, stream=True, **common)
        for frame_idx, r in enumerate(results):
            frame = r.orig_img
            fh, fw = frame.shape[:2]
            dets = []
            if r.boxes is not None and len(r.boxes):
                ids = r.boxes.id
                ids = ids.int().tolist() if ids is not None else [-1] * len(r.boxes)
                for box_xyxy, conf, rid in zip(
                        r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), ids):
                    if not in_roi(box_xyxy, self.cfg.roi):
                        continue
                    bw = box_xyxy[2] - box_xyxy[0]
                    bh = box_xyxy[3] - box_xyxy[1]
                    if bw <= 0 or bh <= 0:
                        continue
                    frac = max(bw, bh) / max(fw, fh)
                    aspect = max(bw / bh, bh / bw)
                    if not (self.cfg.min_size_frac <= frac <= self.cfg.max_size_frac):
                        continue
                    if aspect > self.cfg.max_aspect:
                        continue
                    dets.append(Detection(raw_id=rid, xyxy=tuple(box_xyxy), conf=conf))
            yield frame_idx, frame, dets
