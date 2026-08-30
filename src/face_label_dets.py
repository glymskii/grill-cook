"""Attach a face class (raw/cooked/cheese) to every detection in a dump.

Sequential decode, not seeking: the dump is 5 fps over a 60 fps file, and
seeking 5000 times costs more than reading straight through.

Usage: python src/face_label_dets.py data/mot_dets_full.json data/IMG_6635.mov
"""
import json
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
from face_stats import stats_for                       # noqa: E402

TF = transforms.Compose([
    transforms.Resize((112, 112)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def main():
    dets_p, vid_p = Path(sys.argv[1]), Path(sys.argv[2])
    rows = json.loads(dets_p.read_text())
    with open(ROOT / "data/face_clf.pkl", "rb") as f:
        bundle = pickle.load(f)
    clf = bundle["clf"]
    stats_ref = np.load(ROOT / "data/face_stats.npy")
    mu, sd = stats_ref.mean(0), stats_ref.std(0) + 1e-6

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = torch.nn.Identity()
    net.eval().to(dev)

    cap = cv2.VideoCapture(str(vid_p))
    fps = cap.get(cv2.CAP_PROP_FPS)
    idx, ri, done = 0, 0, 0
    while ri < len(rows):
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        idx += 1
        if t + 1e-6 < rows[ri]["t"]:
            continue
        row = rows[ri]; ri += 1
        fh, fw = frame.shape[:2]
        crops, keep = [], []
        for i, d in enumerate(row["dets"]):
            R = int(d[2] * fw * 1.15)
            cx, cy = int(d[0] * fw), int(d[1] * fh)
            c = frame[max(0, cy - R):cy + R, max(0, cx - R):cx + R]
            if c.shape[0] < 24 or c.shape[1] < 24:
                continue
            crops.append(cv2.resize(c, (96, 96))); keep.append(i)
        faces = [None] * len(row["dets"])
        if crops:
            with torch.no_grad():
                batch = torch.stack([TF(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                                     for c in crops]).to(dev)
                emb = net(batch).cpu().numpy()
            st = (np.array([stats_for(c) for c in crops], np.float32) - mu) / sd
            pred = clf.predict(np.hstack([emb, st * 3.0]))
            for k, i in enumerate(keep):
                faces[i] = int(pred[k])
        row["dets"] = [list(d) + [faces[i]] for i, d in enumerate(row["dets"])]
        done += 1
        if done % 500 == 0:
            print(f"  {t:.0f}с", flush=True)
    cap.release()
    out = dets_p.with_name(dets_p.stem + "_faces.json")
    out.write_text(json.dumps(rows))
    print(f"готово: {out} ({done} кадров)")


if __name__ == "__main__":
    main()
