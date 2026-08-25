"""Cook-time recommendation v0.

Anchored to what we actually measured on a real industrial griddle (10 cm,
~1.2 cm patty, ~190 C, cooked from frozen: A 180 s / B 150 s), then scaled by
crude heat-transfer factors. Thickness dominates (between linear and the
conductive h^2), griddle temperature is worth ~0.8%/C, freezer depth ~0.4%/C.
Diameter is deliberately absent: heat flows through the thickness, not the rim.

This is a starting point, not food safety. Ground beef must reach 71 C / 160 F
inside — only a thermometer proves that, and the UI must say so.
"""


def recommend(d_cm: float, h_cm: float, griddle_c: float, freezer_c: float):
    k = (max(0.4, h_cm) / 1.2) ** 1.7
    k *= 1 + max(-60, min(60, 190 - griddle_c)) * 0.008
    k *= 1 + max(0.0, -freezer_c - 18) * 0.004
    clamp = lambda v: round(max(45, min(600, v)))
    return {"side_a": clamp(180 * k), "side_b": clamp(150 * k),
            "note": "v0 estimate — calibrate with a thermometer: ground beef must reach 160°F / 71°C inside."}
