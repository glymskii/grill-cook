"""Synthesise a week of shift-level cooking data for the customer demo.

Every number here is invented. It exists to show what the system reports, not
what any kitchen actually did — the deliverables label it as such on every page.
The distributions are anchored to what we measured on the customer's own griddle
(side A 203 s, side B 151 s), so the shapes are plausible rather than arbitrary.
"""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 20260807
TARGET_A, TARGET_B = 180.0, 150.0      # standard, seconds per side
TOL = 20.0                              # tolerance, seconds

SHIFTS = [("Утро", 7, 15), ("День", 15, 23)]
COOKS = {
    "Утро": ["А. Ким", "Д. Сафин", "М. Егорова"],
    "День": ["Р. Ильясов", "Т. Нурлан", "О. Величко"],
}
# per-cook behaviour: (bias on side A, bias on side B, sloppiness)
STYLE = {
    "А. Ким":       (2, -3, 8),
    "Д. Сафин":     (-14, -10, 16),   # consistently rushes
    "М. Егорова":   (4, 2, 7),
    "Р. Ильясов":   (1, 1, 9),
    "Т. Нурлан":    (22, 14, 19),     # holds patties too long under load
    "О. Величко":   (-2, 3, 10),
}


def verdict(a: float, b: float) -> str:
    if a < TARGET_A - TOL or b < TARGET_B - TOL:
        return "Недожар"
    if a > TARGET_A + TOL or b > TARGET_B + TOL:
        return "Пережар"
    return "Норма"


def main():
    rng = random.Random(SEED)
    start = datetime(2026, 7, 27)
    patties, days = [], []

    for d in range(7):
        day = start + timedelta(days=d)
        weekend = day.weekday() >= 5
        for shift_name, h0, h1 in SHIFTS:
            cooks = COOKS[shift_name]
            for hour in range(h0, h1):
                # lunch and evening rushes, weekends busier
                load = 1.0
                if 12 <= hour <= 14:
                    load = 2.3
                elif 18 <= hour <= 20:
                    load = 2.0
                if weekend:
                    load *= 1.25
                n = max(1, int(rng.gauss(9 * load, 2.5)))
                for _ in range(n):
                    cook = rng.choice(cooks)
                    ba, bb, sd = STYLE[cook]
                    # under rush everyone drifts faster and less consistently
                    rush = (load - 1.0) * 9
                    a = rng.gauss(TARGET_A + ba - rush, sd + rush * 0.5)
                    b = rng.gauss(TARGET_B + bb - rush * 0.7, sd + rush * 0.4)
                    flips = 1 if rng.random() > 0.12 else rng.choice([2, 2, 3])
                    ts = day + timedelta(hours=hour, minutes=rng.uniform(0, 59))
                    patties.append({
                        "id": len(patties) + 1,
                        "ts": ts.isoformat(timespec="minutes"),
                        "date": day.strftime("%Y-%m-%d"),
                        "hour": hour,
                        "shift": shift_name,
                        "cook": cook,
                        "side_a": round(max(30, a), 1),
                        "side_b": round(max(30, b), 1),
                        "flips": flips,
                        "verdict": verdict(a, b),
                    })
        days.append(day.strftime("%Y-%m-%d"))

    ok = sum(1 for p in patties if p["verdict"] == "Норма")
    summary = {
        "generated": "2026-08-07",
        "period": {"from": days[0], "to": days[-1]},
        "target": {"side_a": TARGET_A, "side_b": TARGET_B, "tolerance": TOL},
        "total": len(patties),
        "compliance": round(100 * ok / len(patties), 1),
        "disclaimer": "Иллюстративный пример. Все данные синтетические и не "
                      "относятся к реальному предприятию.",
    }

    out = Path(__file__).parent
    (out / "data.json").write_text(json.dumps(
        {"summary": summary, "patties": patties}, ensure_ascii=False, indent=1))
    print(f"котлет: {len(patties)}   соответствие стандарту: {summary['compliance']}%")
    for name in STYLE:
        sub = [p for p in patties if p["cook"] == name]
        good = sum(1 for p in sub if p["verdict"] == "Норма")
        print(f"  {name:12s} n={len(sub):4d}  норма {100*good/len(sub):5.1f}%")


if __name__ == "__main__":
    main()
