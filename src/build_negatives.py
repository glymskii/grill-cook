"""Rebuild the "other" half of the face dataset from data/faces_other.

Writes exactly the three files the trainer reads, so adding negatives is a
matter of dropping crops into that folder and running this.

Usage: python src/build_negatives.py
"""
import json
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

OTHER = ROOT / "data/faces_other"
TF = transforms.Compose([
    transforms.Resize((112, 112)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def main():
    paths = sorted(OTHER.glob("*.jpg"))
    print(f"негативов: {len(paths)}")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = torch.nn.Identity()
    net.eval().to(dev)
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), 128):
            batch = torch.stack([TF(Image.open(p).convert("RGB"))
                                 for p in paths[i:i + 128]]).to(dev)
            feats.append(net(batch).cpu().numpy())
            print(f"  {i + len(batch)}/{len(paths)}", flush=True)
    np.save(ROOT / "data/other_feats.npy", np.vstack(feats))
    stats = np.array([stats_for(cv2.imread(str(p))) for p in paths], dtype=np.float32)
    np.save(ROOT / "data/other_stats.npy", stats)
    (ROOT / "data/other_paths.json").write_text(json.dumps([p.name for p in paths]))
    kinds = {}
    for p in paths:
        kinds[p.name.split("_")[0]] = kinds.get(p.name.split("_")[0], 0) + 1
    print(f"по источникам: {kinds}")


if __name__ == "__main__":
    main()
