"""Cut the real patty out of the reference cook so it can be re-staged.

Everything here comes from the customer's own footage: the empty griddle, the
patty with its raw face up, the same patty with its seared face up, and the
cheese-covered version. Compositing those real pixels keeps texture, lighting
and grain honest, which a generated clip cannot promise -- and unlike a
generated clip, the timings are ours, so the ground truth is exact.

Usage: python src/extract_sprites.py --out data/sprites
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOWorld

from config import Config

VIDEO = "data/Котлета Жарка.mp4"

# moments to lift a sprite from, chosen off the verified timeline
SAMPLES = {
    "raw":    [20.0, 45.0, 80.0, 120.0, 155.0],   # face B up, pink -> greying
    "seared": [167.9, 168.1, 168.3, 171.1, 217.3],  # face A up, crust
    "cheese": [240.0, 280.0, 320.0],
}
BG_FRAMES = [1.5, 2.5, 3.5, 4.5, 5.5]   # griddle before anything is put on it


def cut_patty(frame, box, feather=4):
    """Segment the patty inside its box; the griddle is far darker, so a
    lightness threshold separates them cleanly."""
    x1, y1, x2, y2 = [int(v) for v in box]
    pad = int(0.10 * max(x2 - x1, y2 - y1))
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = frame[y1:y2, x1:x2]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    _, mask = cv2.threshold(lab[:, :, 0], 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    # Steam is bright too and Otsu happily keeps it, which leaves a pale
    # rectangle around the sprite once it is pasted somewhere else. The patty is
    # the thing filling the detection box, so clip to that ellipse.
    ch, cw = mask.shape
    limit = np.zeros_like(mask)
    cv2.ellipse(limit, (cw // 2, ch // 2),
                (int(cw * 0.44), int(ch * 0.44)), 0, 0, 360, 255, -1)
    mask = cv2.bitwise_and(mask, limit)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(lbl == big, 255, 0).astype(np.uint8)
    # pull the edge inwards before feathering: the griddle immediately around
    # the patty is lit differently from wherever the sprite gets pasted, and any
    # of it left under a soft edge shows up as a rectangular halo
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    mask = cv2.GaussianBlur(mask, (feather * 2 + 1,) * 2, 0)
    ys, xs = np.nonzero(mask > 8)
    if len(ys):
        crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return crop, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sprites")
    ap.add_argument("--video", default=VIDEO)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    model = YOLOWorld(cfg.model_name)
    model.set_classes(list(cfg.classes))
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # clean plate: median over a few pre-cook frames drops drifting steam
    bg = []
    for t in BG_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if ok:
            bg.append(fr)
    background = np.median(np.stack(bg), axis=0).astype(np.uint8)
    cv2.imwrite(str(out / "griddle_bg.png"), background)
    print(f"background from {len(bg)} frames -> {background.shape}")

    for kind, times in SAMPLES.items():
        for i, t in enumerate(times):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, fr = cap.read()
            if not ok:
                continue
            r = model.predict(fr, conf=0.04, imgsz=cfg.imgsz, device=cfg.device,
                              agnostic_nms=True, verbose=False)[0]
            if r.boxes is None or not len(r.boxes):
                print(f"  {kind} t={t}: no detection")
                continue
            j = int(np.argmax(r.boxes.conf.tolist()))
            crop, mask = cut_patty(fr, r.boxes.xyxy.tolist()[j])
            rgba = np.dstack([crop, mask])
            cv2.imwrite(str(out / f"{kind}_{i}.png"), rgba)
            print(f"  {kind}_{i}  t={t:6.1f}  {crop.shape[1]}x{crop.shape[0]}  "
                  f"cover={mask.mean() / 255:.2f}")
    cap.release()


if __name__ == "__main__":
    main()
