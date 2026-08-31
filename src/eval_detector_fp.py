"""Compare detectors by what they invent, not only by what they find.

mAP on auto-labels cannot see this: the labels came from the same detector, so
its favourite grease marks are in the ground truth. Instead each proposal is put
to the face classifier — the share it calls "other" is the false-positive rate
the cook actually sees as a ring on nothing.

Usage: python src/eval_detector_fp.py v6 v7
"""
import pickle
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
from config import Config                            # noqa: E402
from face_stats import stats_for                     # noqa: E402
from pipeline import in_poly                         # noqa: E402

WEIGHTS = {"v6": "runs/detect/runs/detect/patty_v6/weights/best.pt",
           "v7": "runs/detect/runs/detect/patty_v7/weights/best.pt"}
SAMPLES = [("data/IMG_6635.mov", [[0.22,0.72],[0.48,0.09],[0.95,0.20],[0.70,0.97]], 120, 990, 15),
           ("data/IMG_6637.mov", [[0.215,0.70],[0.415,0.085],[0.885,0.165],[0.695,0.92]], 20, 160, 5)]
TF = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])


def main():
    cfg = Config()
    from ultralytics import YOLO
    with open(ROOT / "data/face_clf.pkl", "rb") as f:
        clf = pickle.load(f)["clf"]
    ref = np.load(ROOT / "data/face_stats.npy"); mu, sd = ref.mean(0), ref.std(0) + 1e-6
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = torch.nn.Identity(); net.eval().to(cfg.device)

    for tag in sys.argv[1:]:
        det = YOLO(str(ROOT / WEIGHTS[tag]))
        tot = other = weak = 0
        confs = []
        for vid, roi, t0, t1, step in SAMPLES:
            cap = cv2.VideoCapture(str(ROOT / vid)); fps = cap.get(cv2.CAP_PROP_FPS)
            for t in range(t0, t1, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
                ok, fr = cap.read()
                if not ok:
                    continue
                fh, fw = fr.shape[:2]
                r = det.predict(fr, conf=0.15, iou=cfg.iou, imgsz=960,
                                device=cfg.device, agnostic_nms=True, verbose=False)[0]
                crops = []
                for (x1, y1, x2, y2), cf in zip(
                        [] if r.boxes is None else r.boxes.xyxy.tolist(),
                        [] if r.boxes is None else r.boxes.conf.tolist()):
                    if not in_poly((x1+x2)/2/fw, (y1+y2)/2/fh, roi):
                        continue
                    R = int(max(x2-x1, y2-y1) * 0.575)
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                    c = fr[max(0,cy-R):cy+R, max(0,cx-R):cx+R]
                    if c.shape[0] < 24 or c.shape[1] < 24:
                        continue
                    crops.append(cv2.resize(c, (96, 96))); confs.append(cf)
                    weak += cf < 0.5
                if not crops:
                    continue
                with torch.no_grad():
                    b = torch.stack([TF(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                                     for c in crops]).to(cfg.device)
                    emb = net(b).cpu().numpy()
                st = (np.array([stats_for(c) for c in crops], np.float32) - mu) / sd
                pred = clf.predict(np.hstack([emb, st * 3.0]))
                tot += len(pred); other += int((pred == 3).sum())
            cap.release()
        print(f"{tag}: предложений {tot}, из них 'не котлета' {other} "
              f"({100*other/max(tot,1):.1f}%), слабых (<0.5) {weak} "
              f"({100*weak/max(tot,1):.1f}%), медиана conf {np.median(confs):.2f}")


if __name__ == "__main__":
    main()
