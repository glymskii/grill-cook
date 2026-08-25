"""Render manually verified cooking events over the original industrial-kitchen video."""
import cv2

SRC = "out/industrial_real/input_10fps.mp4"
OUT = "out/industrial_real/original_verified_720p.mp4"

PLACED = 8.5
LIFTED_FOR_FLIP = 166.5
SIDE2_CONTACT = 171.0
REMOVED = 363.5


def fmt(seconds: float) -> str:
    seconds = max(0.0, seconds)
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


cap = cv2.VideoCapture(SRC)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (720, 1280))

while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    frame = cv2.resize(frame, (720, 1280))

    if t < PLACED:
        state, side1, side2 = "WAITING", 0.0, 0.0
    elif t < LIFTED_FOR_FLIP:
        state, side1, side2 = "SIDE 1 ON GRIDDLE", t - PLACED, 0.0
    elif t < SIDE2_CONTACT:
        state, side1, side2 = "FLIPPING / NO CONTACT", LIFTED_FOR_FLIP - PLACED, 0.0
    elif t < REMOVED:
        state = "SIDE 2 ON GRIDDLE"
        side1, side2 = LIFTED_FOR_FLIP - PLACED, t - SIDE2_CONTACT
    else:
        state = "REMOVED"
        side1, side2 = LIFTED_FOR_FLIP - PLACED, REMOVED - SIDE2_CONTACT

    # Header panel.
    cv2.rectangle(frame, (15, 15), (705, 170), (12, 12, 12), -1)
    cv2.rectangle(frame, (15, 15), (705, 170), (40, 220, 255), 3)
    cv2.putText(frame, "TRACK ID: 1", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 220, 255), 2)
    cv2.putText(frame, state, (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(frame, f"SIDE 1: {fmt(side1)}", (35, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 220, 90), 2)
    cv2.putText(frame, f"SIDE 2: {fmt(side2)}", (370, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 170, 255), 2)

    # Fixed-camera ROI for the patty; hidden while it is lifted/removed.
    if PLACED <= t < LIFTED_FOR_FLIP or SIDE2_CONTACT <= t < REMOVED:
        cv2.rectangle(frame, (295, 700), (500, 855), (40, 220, 255), 3)
        cv2.putText(frame, "#1", (300, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 220, 255), 2)

    writer.write(frame)

cap.release()
writer.release()
