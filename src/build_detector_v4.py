"""Detector dataset v4 = v3 + what the v9 review found the detector blind or wrong on.

Three additions, each traced to a defect in the reels:

  1. The grate edge beside a patty (smash 95-165 s) - the detector fires there in
     87% of frames and the engine had to invent rules against it. Frames from
     that window go in with boxes for the real patties only; the grate becomes
     background.
  2. Places the review saw empty on the long shift (short bare slots) - same
     treatment, at their own moments.
  3. Freshly pressed smash patties (8-24 s) - the detector produced nothing for
     up to 13 s after the press. Boxes come from the slot anchors, which
     back-dating proved occupied from the moment the meat landed; only frames
     where the anchor crop is meat (a* >= 135 - pink, not the grey press) are used.

Real patties are boxed the v3 way (detector proposals cleaned by the face
classifier), so the new frames carry the same label geometry as the rest.

Usage: python src/build_detector_v4.py        -> data/detector_v4 + trains patty_v8
"""
import json
import pickle
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
from config import Config                                # noqa: E402
from face_stats import stats_for                         # noqa: E402

SRC = ROOT / "data/detector_v3"
OUT = ROOT / "data/detector_v4"
TF = transforms.Compose([
    transforms.Resize((112, 112)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# (video, t0, t1, step, places to force-empty [(x, y, r)], anchor boxes [(x, y, r, t_from, t_to)])
JOBS = [
    ("data/IMG_6637.mov", 95.0, 165.0, 2.0, [(0.327, 0.484, 0.06)], []),
    ("data/IMG_6637.mov", 8.0, 24.0, 0.5, [],
     [(0.378, 0.521, 0.057, 8.0, 24.0), (0.611, 0.594, 0.081, 14.0, 24.0),
      (0.453, 0.373, 0.065, 18.0, 24.0)]),
]


def main():
    cfg = Config()
    from ultralytics import YOLO
    det = YOLO(str(ROOT / "runs/detect/runs/detect/patty_v7/weights/best.pt"))
    with open(ROOT / "data/face_clf.pkl", "rb") as f:
        clf = pickle.load(f)["clf"]
    ref = np.load(ROOT / "data/face_stats.npy")
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = torch.nn.Identity(); net.eval().to(cfg.device)

    # short bare slots of the long shift become force-empty places at their times
    life = json.loads((ROOT / "out/v9_review/life_long.json").read_text())
    gt = json.loads((ROOT / "data/flip_gt_slots_long.json").read_text())
    for sid in gt.get("not_a_patty", []):
        s = life.get(str(sid))
        if s:
            x, y, r = s["anchor"]
            JOBS.append(("data/IMG_6635.mov", s["start"], s["start"] + s["life"], 2.0,
                         [(x, y, r)], []))

    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / f"images/{split}").mkdir(parents=True)
        (OUT / f"labels/{split}").mkdir(parents=True)
        for f in (SRC / f"images/{split}").iterdir():
            shutil.copy(f, OUT / f"images/{split}" / f.name)
        for f in (SRC / f"labels/{split}").iterdir():
            shutil.copy(f, OUT / f"labels/{split}" / f.name)
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: patty\n")

    def face_class(frame, x1, y1, x2, y2):
        crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        if crop.size == 0:
            return 3
        with torch.no_grad():
            feat = net(TF(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                       .unsqueeze(0).to(cfg.device)).cpu().numpy()
        st_ = ((np.array(stats_for(crop), dtype=np.float32) - mu) / sd) * 3.0
        return int(clf.predict(np.hstack([feat, st_[None]]))[0])

    added = {"train": [0, 0], "val": [0, 0]}          # frames, boxes
    forced_empty, anchor_boxes = 0, 0
    for vid, t0, t1, step, empties, anchors in JOBS:
        cap = cv2.VideoCapture(str(ROOT / vid))
        fps = cap.get(cv2.CAP_PROP_FPS)
        stem = Path(vid).stem
        t = t0
        while t <= t1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            if not ok:
                break
            fh, fw = frame.shape[:2]
            boxes = []
            r = det.predict(frame, conf=0.25, iou=cfg.iou, imgsz=960, device=cfg.device,
                            agnostic_nms=True, verbose=False)[0]
            if r.boxes is not None:
                for (x1, y1, x2, y2), cf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                    cx, cy = (x1 + x2) / 2 / fw, (y1 + y2) / 2 / fh
                    if any(np.hypot(cx - ex, cy - ey) < 0.9 * er for ex, ey, er in empties):
                        forced_empty += 1
                        continue                      # this place is background now
                    if face_class(frame, x1, y1, x2, y2) == 3:
                        continue
                    boxes.append((cx, cy, (x2 - x1) / fw, (y2 - y1) / fh))
            lab_img = None
            for ax, ay, ar, tf, tt in anchors:
                if not (tf <= t <= tt):
                    continue
                if lab_img is None:
                    lab_img = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                R = int(ar * fw * 0.6)
                cxp, cyp = int(ax * fw), int(ay * fh)
                patch = lab_img[max(0, cyp - R):cyp + R, max(0, cxp - R):cxp + R]
                if patch.size and patch.reshape(-1, 3).mean(0)[1] >= 135:     # pink meat, not the press
                    if not any(np.hypot(ax - bx, ay - by) < 0.5 * ar for bx, by, _, _ in boxes):
                        boxes.append((ax, ay, 2 * ar, 2 * ar * fw / fh))
                        anchor_boxes += 1
            # alternating 20-s blocks keep val honest, as in v3
            split = "val" if int(t // 20) % 3 == 2 else "train"
            name = f"{stem}_v4_{int(t * 10):06d}"
            cv2.imwrite(str(OUT / f"images/{split}/{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(OUT / f"labels/{split}/{name}.txt", "w") as f:
                for cx, cy, w, h in boxes:
                    f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            added[split][0] += 1
            added[split][1] += len(boxes)
            t += step
        cap.release()
    print(f"v4: added {added}; proposals forced to background {forced_empty}; "
          f"anchor boxes for fresh smash {anchor_boxes}")
    print("training patty_v8 from v7, 25 epochs")
    model = YOLO(str(ROOT / "runs/detect/runs/detect/patty_v7/weights/best.pt"))
    model.train(data=str(OUT / "data.yaml"), epochs=25, imgsz=960, batch=8,
                device=cfg.device, project=str(ROOT / "runs/detect/runs/detect"),
                name="patty_v8", exist_ok=True, verbose=False, plots=False)


if __name__ == "__main__":
    main()
