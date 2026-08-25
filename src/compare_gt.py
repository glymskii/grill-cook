"""Compare pipeline events against manual ground truth.

Usage: python compare_gt.py out/run2/events.json data/frozen_patties_gt.json --offset 4.0
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("gt")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="clip start offset: source_t = clip_t + offset")
    args = ap.parse_args()

    data = json.load(open(args.events))
    gt = json.load(open(args.gt))

    print("scene cuts (source t):",
          [round(c + args.offset, 1) for c in data.get("scene_cuts", [])])
    print("\npipeline events (source t):")
    by_patty = {}
    for e in data["events"]:
        t = round(e["t"] + args.offset, 1)
        by_patty.setdefault(e["patty"], []).append(f"{e['type']}@{t}"
            + (f"(dE{e['delta_e']})" if "delta_e" in e else ""))
    for pid, evs in sorted(by_patty.items()):
        print(f"  patty {pid}: " + "  ".join(evs))

    print("\nground truth:")
    for name, g in gt["patties"].items():
        flips = "  ".join(f"FLIP@{t}" for t in g["flips"])
        extra = f"  FLIP3?@{g['flip3_uncertain']}" if g.get("flip3_uncertain") else ""
        print(f"  {name}: PLACED@{g['placed']}  {flips}{extra}  REMOVED@{g['removed']}")


if __name__ == "__main__":
    main()
