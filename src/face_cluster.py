"""Cluster patty crops and render one montage per cluster for labeling.

Hand-labeling thousands of crops is not on; judging a montage of a cluster is.
Frozen ImageNet features separate raw meat, seared crust and cheese well enough
that a cluster is nearly pure, so one verdict labels hundreds of crops.

Usage: python src/face_cluster.py                       # patty faces, K=10
       python src/face_cluster.py data/neg_candidates 12
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
# Defaults cluster the patty faces; pointed at another folder (the mined
# negatives, say) the same machinery labels those the same cheap way.
FACES = ROOT / Path(sys.argv[1] if len(sys.argv) > 1 else "data/faces")
OUT = ROOT / "out" / (FACES.name + "_clusters")
K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
STEM = "face" if FACES.name == "faces" else FACES.name

tf = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def embed(paths, dev):
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = torch.nn.Identity()
    net.eval().to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), 128):
            batch = torch.stack([tf(Image.open(p).convert("RGB"))
                                 for p in paths[i:i + 128]]).to(dev)
            out.append(net(batch).cpu().numpy())
            print(f"  {i + len(batch)}/{len(paths)}", flush=True)
    return np.vstack(out)


def main():
    paths = sorted(FACES.glob("*.jpg"))
    print(f"кропов: {len(paths)}")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    feats = embed(paths, dev)
    np.save(ROOT / f"data/{STEM}_feats.npy", feats)
    (ROOT / f"data/{STEM}_paths.json").write_text(json.dumps([p.name for p in paths]))

    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(feats)
    labels = km.labels_
    (ROOT / f"data/{STEM}_clusters.json").write_text(
        json.dumps({p.name: int(l) for p, l in zip(paths, labels)}))
    OUT.mkdir(parents=True, exist_ok=True)
    for c in range(K):
        idx = np.where(labels == c)[0]
        pick = idx[np.linspace(0, len(idx) - 1, min(48, len(idx))).astype(int)]
        tiles = [cv2.resize(cv2.imread(str(paths[i])), (96, 96)) for i in pick]
        while len(tiles) % 8:
            tiles.append(np.zeros((96, 96, 3), np.uint8))
        rows = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
        img = np.vstack(rows)
        cv2.putText(img, f"cluster {c}  n={len(idx)}", (8, 22), 0, 0.7, (60, 255, 90), 2)
        cv2.imwrite(str(OUT / f"cluster_{c}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"кластеры: {[int((labels==c).sum()) for c in range(K)]}")
    print(f"монтажи: {OUT}")


if __name__ == "__main__":
    main()
