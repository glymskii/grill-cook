"""The face classifier as a callable the slot engine can hold.

The engine stays free of torch: it is handed a function from crops to class ids
and knows nothing else about it. Batching matters — one forward for every slot
on the griddle, not one per slot.

    from face_reader import FaceReader
    eng = SlotEngine(targets, face_model=FaceReader())
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
from face_stats import stats_for                       # noqa: E402

TF = transforms.Compose([
    transforms.Resize((112, 112)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class FaceReader:
    def __init__(self, model=ROOT / "data/face_clf.pkl"):
        with open(model, "rb") as f:
            bundle = pickle.load(f)
        self.clf = bundle["clf"]
        self.classes = bundle["classes"]
        ref = np.load(ROOT / "data/face_stats.npy")
        self.mu, self.sd = ref.mean(0), ref.std(0) + 1e-6
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        net.fc = torch.nn.Identity()
        self.net = net.eval().to(self.dev)

    def __call__(self, crops):
        if not crops:
            return []
        with torch.no_grad():
            batch = torch.stack([
                TF(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                for c in crops]).to(self.dev)
            feats = self.net(batch).cpu().numpy()
        stats = np.array([stats_for(c) for c in crops], dtype=np.float32)
        X = np.hstack([feats, ((stats - self.mu) / self.sd) * 3.0])
        return self.clf.predict(X).tolist()
