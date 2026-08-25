"""Stage a 15-patty cook on the customer's own griddle.

Real pixels, scripted timings. The background, the patty faces and the cheese all
come from data/Котлета Жарка.mp4, so texture, lighting and sensor grain are the
ones the production camera actually produces. What is synthetic is only the
choreography -- which means the ground truth is exact rather than hand-guessed,
and any disagreement with the pipeline is the pipeline's fault.

Perspective is fitted to two real observations from that video: the patty
measured 230 px across at y=1050 and 270 px at y=1186.

Usage: python src/make_composite.py --out data/composite15.mp4 --gt data/composite15_gt.json
"""
import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

W, H = 1080, 1920
FPS = 25
DURATION = 90.0

# diameter as a function of depth, from the two measured points.
# SCALE shrinks the patty: fifteen of the original 14 cm ones simply do not fit
# in this camera's view of the griddle, so the scene uses slider-sized patties.
D_AT_Y, Y_REF, D_SLOPE, SCALE = 230.0, 1050.0, 0.294, 0.78


def diameter_at(y: float) -> float:
    return (D_AT_Y + (y - Y_REF) * D_SLOPE) * SCALE


# The griddle's back edge runs diagonally: it sits at y=1060 on the left of the
# frame and rises to y=560 on the right. Rows are laid parallel to it so every
# patty lands on cooking surface rather than on the rail or the drip tray.
def back_edge(x: float) -> float:
    return 1060.0 - 0.50 * x


# Rows sit at constant image height, because on a flat surface that means
# constant distance from the camera and therefore constant apparent size. Rows
# further back are shorter: the griddle's diagonal edge eats into them.
# (y, count, x_start, x_end)
LAYOUT = [
    (880, 3, 620, 1010),
    (1040, 4, 320, 980),
    (1200, 5, 120, 980),
    (1380, 3, 170, 890),
]


def build_slots():
    slots = []
    for row, (y, count, x0, x1) in enumerate(LAYOUT):
        for i in range(count):
            f = i / max(1, count - 1)
            x = x0 + (x1 - x0) * f
            # stagger alternate patties in depth so a row never forms a
            # perfectly regular line of touching discs
            slots.append((x, y + (28 if i % 2 else -28)))
    return slots


def paste(dst, sprite_bgra, cx, cy, target_d, angle, gain):
    """Alpha-composite a sprite centred at (cx, cy), sized to target_d."""
    bgr = sprite_bgra[:, :, :3].astype(np.float32) * gain
    alpha = sprite_bgra[:, :, 3].astype(np.float32) / 255.0
    # steepen the edge: a long alpha ramp drags the sprite's own griddle
    # pixels into the destination and reads as a pale rectangle
    alpha = np.clip((alpha - 0.42) / 0.28, 0.0, 1.0)
    h, w = alpha.shape
    scale = target_d / max(w, h)
    nw, nh = max(4, int(w * scale)), max(4, int(h * scale))
    bgr = cv2.resize(bgr, (nw, nh))
    alpha = cv2.resize(alpha, (nw, nh))

    m = cv2.getRotationMatrix2D((nw / 2, nh / 2), angle, 1.0)
    bgr = cv2.warpAffine(bgr, m, (nw, nh), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
    alpha = cv2.warpAffine(alpha, m, (nw, nh), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)

    x1, y1 = int(cx - nw / 2), int(cy - nh / 2)
    x2, y2 = x1 + nw, y1 + nh
    sx1, sy1 = max(0, -x1), max(0, -y1)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    sub_a = alpha[sy1:sy1 + (y2 - y1), sx1:sx1 + (x2 - x1)][..., None]
    sub_c = bgr[sy1:sy1 + (y2 - y1), sx1:sx1 + (x2 - x1)]
    region = dst[y1:y2, x1:x2].astype(np.float32)
    dst[y1:y2, x1:x2] = np.clip(region * (1 - sub_a) + sub_c * sub_a, 0, 255).astype(np.uint8)
    return (x1, y1, x2, y2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sprites", default="data/sprites")
    ap.add_argument("--out", default="data/composite15.mp4")
    ap.add_argument("--gt", default="data/composite15_gt.json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sp = Path(args.sprites)
    bg = cv2.imread(str(sp / "griddle_bg.png"))
    raws = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in sorted(sp.glob("raw_*.png"))]
    seareds = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in sorted(sp.glob("seared_*.png"))]
    cheeses = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in sorted(sp.glob("cheese_*.png"))]

    slots = build_slots()
    assert len(slots) == 15, len(slots)

    # Timings are staggered and jittered per patty: a uniform schedule would let
    # a broken pipeline look right by reporting one constant for everybody.
    patties = []
    for i, (x, y) in enumerate(slots):
        t_place = 2.0 + 1.15 * i + rng.uniform(-0.2, 0.2)
        t_flip = 32.0 + 1.15 * i + rng.uniform(-2.5, 2.5)
        t_remove = 68.0 + 1.15 * i + rng.uniform(-2.0, 2.0)
        patties.append({
            "id": i + 1, "x": x, "y": y, "d": diameter_at(y),
            "t_place": round(t_place, 2), "t_flip": round(t_flip, 2),
            "t_remove": round(t_remove, 2),
            "raw": rng.randrange(len(raws)), "seared": rng.randrange(len(seareds)),
            "cheese": rng.randrange(len(cheeses)),
            "with_cheese": i % 3 == 0,          # five of them get a slice
            "t_cheese": round(t_flip + rng.uniform(3.0, 6.0), 2),
            # only a small tilt: the sprite carries the camera's foreshortening,
            # so spinning it far would put the patty in a plane of its own
            "angle": rng.uniform(-22, 22), "gain": rng.uniform(0.92, 1.08),
        })

    hide = 0.36    # spatula covers the patty this long during a flip
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n_frames = int(DURATION * FPS)
    for f in range(n_frames):
        t = f / FPS
        frame = bg.copy()
        # gentle global flicker + grain so the plate is never pixel-identical
        frame = np.clip(frame.astype(np.float32)
                        * (1.0 + 0.012 * math.sin(t * 1.7))
                        + np.random.normal(0, 1.6, frame.shape), 0, 255).astype(np.uint8)

        for p in sorted(patties, key=lambda q: q["y"]):   # far to near
            if t < p["t_place"] or t >= p["t_remove"]:
                continue
            if p["t_flip"] <= t < p["t_flip"] + hide:      # in the cook's hand
                continue
            if t >= p["t_flip"] + hide:
                spr = (cheeses[p["cheese"]] if p["with_cheese"] and t >= p["t_cheese"]
                       else seareds[p["seared"]])
            else:
                spr = raws[p["raw"]]
            paste(frame, spr, p["x"], p["y"], p["d"], p["angle"], p["gain"])

        writer.write(frame)
        if f % 250 == 0:
            print(f"  t={t:5.1f}s", flush=True)
    writer.release()

    gt = {
        "video": args.out, "fps": FPS, "duration_s": DURATION,
        "note": "Скомпонован из data/Котлета Жарка.mp4: реальный фон и реальные "
                "спрайты котлеты, тайминги заданы скриптом. Сторона A лежит на "
                "плите от укладки до переворота, сторона B — от переворота до снятия.",
        "patties": [{
            "id": p["id"], "x": round(p["x"]), "y": round(p["y"]),
            "placed_t": p["t_place"], "flip_t": p["t_flip"], "removed_t": p["t_remove"],
            "cheese_t": p["t_cheese"] if p["with_cheese"] else None,
            "side_A_s": round(p["t_flip"] - p["t_place"], 2),
            "side_B_s": round(p["t_remove"] - p["t_flip"] - hide, 2),
            "total_s": round(p["t_remove"] - p["t_place"], 2),
        } for p in patties],
    }
    Path(args.gt).write_text(json.dumps(gt, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out} and {args.gt}")


if __name__ == "__main__":
    main()
