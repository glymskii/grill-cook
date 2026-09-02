"""The other half of the scorecard: slots that should not exist, rings that come late.

Flip recall/precision (flip_gt_slots.py score) says nothing about a ring on bare
plate. This reads the life file a slot_life run leaves behind and reports what
the cook would notice: how many slots never held a patty, how long meat sat on
the plate before its ring appeared, and whether the known false places from the
v9 review are still occupied by a slot.

  python src/slot_health.py smash|long
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# places the v9 review established as not-a-patty (x, y), per session
FALSE_PLACES = {"smash": [(0.328, 0.483)],
                "long": [(0.0, 0.0)]}          # filled from the review's short bare slots below
TAG = sys.argv[1]
life = json.loads((ROOT / f"out/v9_review/life_{TAG}_new.json").read_text())
old = json.loads((ROOT / f"out/v9_review/life_{TAG}.json").read_text())
gt = json.loads((ROOT / f"data/flip_gt_slots_{TAG}.json").read_text())

bare_ids = gt.get("not_a_patty", [])
false_places = [(tuple(old[str(i)]["anchor"][:2]), old[str(i)]["start"], old[str(i)]["start"] + old[str(i)]["life"])
                for i in bare_ids if str(i) in old]
if TAG == "smash":
    false_places = [(FALSE_PLACES["smash"][0], 62.0, 165.0)]

n = len(life)
short = [sid for sid, s in life.items() if s["life"] < 70]
gaps = [s["born"] - s["start"] for s in life.values()]
still_false = []
for (px, py), t0, t1 in false_places:
    for sid, s in life.items():
        overlap = min(t1, s["start"] + s["life"]) - max(t0, s["start"])
        if math.hypot(s["anchor"][0] - px, s["anchor"][1] - py) <= 0.03 and overlap >= 10:
            still_false.append((sid, round(overlap)))
print(f"{TAG}: {n} slots (was {len(old)});  short-lived <70s: {len(short)} {short}")
print(f"  seconds on the plate before a ring: p50 {st.median(gaps):.1f}  max {max(gaps):.1f}  "
      f"over 5s: {sum(1 for g in gaps if g > 5)}")
print(f"  slots on places the review saw empty, overlapping that window >=10s (sid, seconds): {still_false or 'none'}")
