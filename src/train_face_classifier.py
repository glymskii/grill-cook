"""Train the patty-face classifier.

Mean colour cannot tell the two faces apart once both have greyed, so this reads
texture instead: ImageNet ResNet18 features feeding a logistic regression. With
roughly a hundred face-A crops a frozen backbone plus a linear head is the right
size of model — fine-tuning would just memorise them.

Evaluation splits by episode, never at random. Crops 40 ms apart are near
duplicates, so a random split would put copies of the same instant on both sides
and report a score the model has not earned. The held-out episodes are the ones
the previous algorithm actually got wrong: the 3:36 flip (A_3) and the long
173-216 s stretch (B_late).

Usage: python src/train_face_classifier.py --data data/faces --out models/face_clf.joblib
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import models, transforms

TEST_EPISODES = {"A_3", "B_late", "H_4", "H_5"}
TOPPING_TEST_FROM = 300.0   # last third of the cheese stretch is held out


def build_backbone(device):
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net.eval().to(device)
    return net


NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
BASE = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                           transforms.ToTensor(), NORM])

# A face keeps its texture but not its colour: the raw side greys steadily as it
# sits over the heat, so tone is a shortcut that stops working mid-cook. These
# transforms strip that shortcut out of the training set and leave the crust
# pattern as the only thing left to learn from.
AUG = transforms.Compose([
    transforms.ToPILImage(),
    # a patty far from the camera arrives as a handful of pixels blown back up,
    # which softens it; without this the model reads small patties as motion blur
    # and files them under HANDLING
    transforms.RandomApply([transforms.Resize((84, 84))], p=0.15),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=12, translate=(0.05, 0.05), scale=(0.92, 1.08)),
    transforms.ColorJitter(brightness=0.45, contrast=0.35, saturation=0.6, hue=0.08),
    transforms.RandomGrayscale(p=0.35),
    transforms.ToTensor(), NORM,
])


def embed(paths, net, device, bs=64, augment=0, seed=0):
    """Embed crops; augment>0 adds that many jittered copies of each."""
    torch.manual_seed(seed)
    variants = [BASE] + [AUG] * augment
    feats, order = [], []
    for vi, tf in enumerate(variants):
        with torch.no_grad():
            for i in range(0, len(paths), bs):
                batch = []
                for p in paths[i:i + bs]:
                    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                    batch.append(tf(img))
                x = torch.stack(batch).to(device)
                feats.append(net(x).cpu().numpy())
        order.extend(range(len(paths)))
    return np.concatenate(feats), np.array(order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/faces")
    ap.add_argument("--out", default="models/face_clf.joblib")
    ap.add_argument("--augment", type=int, default=6,
                    help="jittered copies per training crop")
    ap.add_argument("--drop", default="",
                    help="comma-separated classes to exclude entirely")
    args = ap.parse_args()

    index = json.loads((Path(args.data) / "index.json").read_text())
    # "being handled" is a property of the track, not of the picture: the
    # detector already loses or displaces the box when a patty is lifted. Keeping
    # it as a fourth class leaves a small, vague category that balanced weighting
    # inflates until it swallows every ambiguous face.
    dropped = {c.strip() for c in args.drop.split(",") if c.strip()}
    if dropped:
        index = [r for r in index if r["label"] not in dropped]
        print(f"dropped classes {sorted(dropped)} -> {len(index)} crops left")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = build_backbone(device)

    paths = [r["path"] for r in index]
    labels = np.array([r["label"] for r in index])
    episodes = np.array([r["episode"] for r in index])
    times = np.array([r["t"] for r in index])

    is_test = np.array([
        ep in TEST_EPISODES or (ep == "T_1" and t >= TOPPING_TEST_FROM)
        for ep, t in zip(episodes, times)
    ])
    tr_paths = [p for p, m in zip(paths, is_test) if not m]
    te_paths = [p for p, m in zip(paths, is_test) if m]

    print(f"embedding on {device}: {len(tr_paths)} train (x{1 + args.augment}) "
          f"+ {len(te_paths)} test ...")
    Xtr_raw, idx_tr = embed(tr_paths, net, device, augment=args.augment)
    ytr = labels[~is_test][idx_tr]
    Xte, _ = embed(te_paths, net, device)
    yte = labels[is_test]

    print("train classes:", dict(zip(*np.unique(ytr, return_counts=True))))
    print("test  classes:", dict(zip(*np.unique(yte, return_counts=True))))

    clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced")
    clf.fit(Xtr_raw, ytr)

    pred = clf.predict(Xte)
    print("\n=== HELD-OUT EPISODES ===")
    print(classification_report(yte, pred, digits=3, zero_division=0))
    order = sorted(set(labels))
    print("confusion (rows = truth, cols = pred):", order)
    print(confusion_matrix(yte, pred, labels=order))

    te_eps = episodes[is_test]
    for ep in sorted(set(te_eps)):
        m = te_eps == ep
        acc = (pred[m] == yte[m]).mean()
        print(f"  {ep:8s} n={m.sum():4d}  accuracy {acc:.3f}")

    # ship a model trained on everything; the score above is what it is worth
    Xall, idx_all = embed(paths, net, device, augment=args.augment)
    final = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced")
    final.fit(Xall, labels[idx_all])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({"clf": final, "classes": list(final.classes_)}, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
