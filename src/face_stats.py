"""Hand-picked colour/texture stats to append to the frozen ImageNet features.

A melted cheese slice and a pale chicken face land in the same corner of the
ResNet embedding: both are big, bright and low-contrast. They are not the same
under a ruler — cheese is a smooth yellow sheet, chicken keeps mince texture —
so the classifier gets those numbers explicitly.
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def stats_for(img) -> list[float]:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    sob = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    return [
        *lab.reshape(-1, 3).mean(0), *lab.reshape(-1, 3).std(0),
        float(lap.var()), float(np.abs(sob).mean()),
        float(hsv[..., 1].mean()), float(hsv[..., 1].std()),
        float((g > g.mean() + 20).mean()), float((g < g.mean() - 20).mean()),
    ]


def main():
    names = json.loads((ROOT / "data/face_paths.json").read_text())
    out = np.array([stats_for(cv2.imread(str(ROOT / "data/faces" / n))) for n in names],
                   dtype=np.float32)
    np.save(ROOT / "data/face_stats.npy", out)
    print(f"статистик: {out.shape}")


if __name__ == "__main__":
    main()
