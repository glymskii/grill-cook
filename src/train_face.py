"""Train the patty-face classifier: raw / cooked / cheese.

Labels come from cluster verdicts (data/face_cluster_labels.json) — one human
judgement per montage labels hundreds of crops, and the mixed clusters are
dropped outright: a vague class cost this project 27 accuracy points last time.

The split is by time, never random: crops seconds apart are near-duplicates and
a random split would report a score the model has not earned.

Usage: python src/train_face.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["raw", "cooked", "cheese", "other"]


def main():
    feats = np.load(ROOT / "data/face_feats.npy")
    stats = np.load(ROOT / "data/face_stats.npy")
    stats = (stats - stats.mean(0)) / (stats.std(0) + 1e-6)
    feats = np.hstack([feats, stats * 3.0])   # weight the explicit cues up
    names = json.loads((ROOT / "data/face_paths.json").read_text())
    clusters = json.loads((ROOT / "data/face_clusters.json").read_text())
    lab_map = json.loads((ROOT / "data/face_cluster_labels.json").read_text())
    cluster_to_class = {}
    for cls in CLASSES:
        for c in lab_map.get(cls, []):
            cluster_to_class[c] = cls

    ofeats = np.load(ROOT / "data/other_feats.npy")
    ostats = np.load(ROOT / "data/other_stats.npy")
    onames = json.loads((ROOT / "data/other_paths.json").read_text())
    # normalise the extra cues with the SAME reference the patty crops used
    ostats = (ostats - np.load(ROOT / "data/face_stats.npy").mean(0)) / (
        np.load(ROOT / "data/face_stats.npy").std(0) + 1e-6)
    ofeats = np.hstack([ofeats, ostats * 3.0])

    X, y, t, src = [], [], [], []
    for i, n in enumerate(onames):
        parts = n.split("_")
        X.append(ofeats[i]); y.append(CLASSES.index("other"))
        t.append(int(parts[1]) / 10.0); src.append("long")
    for i, n in enumerate(names):
        cls = cluster_to_class.get(clusters[n])
        if cls is None:
            continue
        tag, ts, _ = n.split("_")
        X.append(feats[i]); y.append(CLASSES.index(cls))
        t.append(int(ts) / 10.0); src.append(tag)
    X, y, t, src = np.array(X), np.array(y), np.array(t), np.array(src)
    print(f"размечено {len(y)} кропов из {len(names)} "
          f"(выброшено смешанных {len(names) - len(y)})")
    for i, c in enumerate(CLASSES):
        print(f"  {c}: {(y == i).sum()}")

    # Split by 60-second blocks, alternating: neighbouring crops stay together
    # (no leakage between near-duplicates) while both halves still cover the
    # whole shift. A plain 70/30 time cut handed every late-session cheese crop
    # to validation and the model was judged on an appearance it never saw.
    block = (t // 60).astype(int)
    train = (block % 2) == 0
    val = ~train
    print(f"\ntrain {train.sum()} / val {val.sum()} "
          f"(чередование 60-с блоков — обе половины покрывают всю смену)")
    for i, c in enumerate(CLASSES):
        print(f"  {c}: train {(y[train] == i).sum()} / val {(y[val] == i).sum()}")

    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(X[train], y[train])
    pred = clf.predict(X[val])
    print("\n" + classification_report(y[val], pred, target_names=CLASSES, digits=3))
    print("матрица ошибок (строки — истина):")
    print(confusion_matrix(y[val], pred))

    import pickle
    with open(ROOT / "data/face_clf.pkl", "wb") as f:
        pickle.dump({"clf": clf, "classes": CLASSES}, f)
    print("\nмодель: data/face_clf.pkl")


if __name__ == "__main__":
    main()
