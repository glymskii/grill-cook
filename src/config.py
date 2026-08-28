"""Configuration for the patty side-timer pipeline."""
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- detection ---
    model_name: str = "yolov8l-worldv2.pt"
    classes: tuple = ("food", "meat", "fried meat patty",
                      "round slice of raw minced meat")
    # YOLO-World scores patties at 0.03-0.08 here — its noise floor. Between
    # 0.055 and 0.06 the count collapses from 11 boxes to 5, so anything above
    # 0.02 is sampling that cliff rather than choosing a real threshold.
    conf: float = 0.02
    iou: float = 0.5
    imgsz: int = 960
    device: str = "mps"
    tracker: str = "src/bytetrack_patty.yaml"
    min_size_frac: float = 0.07   # bbox bigger side vs frame bigger side
    max_size_frac: float = 0.38   # rejects group/whole-pan boxes
    max_aspect: float = 2.6       # perspective flattens far patties into ellipses

    # --- scene cuts (jump cuts in edited video) ---
    # measured on the frame border band (outside pan ROI) — static background
    # changes only when the shot is recut/reframed, not when hands move over the pan
    cut_diff_thr: float = 9.0     # mean abs gray diff of border band => hard cut
    cut_suppress_s: float = 1.2   # no flip decisions this long after a cut
    cut_stitch_mult: float = 2.5  # stitch radius multiplier right after a cut

    # --- pan region of interest (x1, y1, x2, y2) in pixels; None = full frame ---
    roi: tuple | None = None

    # --- track stitching (re-associating raw tracker ids after occlusion) ---
    stitch_max_gap_s: float = 8.0       # patty may vanish this long during a flip
    # a lifted patty travels far in image space (it rises toward the camera),
    # so the stitch radius must cover a full lift-and-place, not just a slide
    stitch_max_dist_diam: float = 2.2   # max center distance to stitch, in patty diameters
    min_track_len_s: float = 1.0        # shorter logical tracks are noise, dropped
    dup_iou: float = 0.45               # overlap above which a box is a duplicate of a tracked patty

    # --- appearance / flip detection ---
    feat_inner_frac: float = 0.6        # central ellipse fraction of bbox used for color
    recent_win_s: float = 0.6           # "after" window for change-point test
    baseline_win_s: float = 2.0         # "before" window
    baseline_gap_s: float = 0.8         # gap between windows (the flip transient itself)
    flip_delta_e: float = 11.0          # Lab delta-E threshold for appearance step
    flip_gap_delta_e: float = 8.0       # lower threshold when a track gap/jump co-occurred
    # both faces of a patty are meat, so a flip changes colour only so much;
    # a bigger jump means something was laid ON the patty (cheese, sauce)
    topping_delta_e: float = 30.0       # above this the step is a topping, not a flip
    # one flip takes several seconds of tumbling, during which appearance keeps
    # changing; the cooldown must outlast the manoeuvre so it counts once
    flip_cooldown_s: float = 12.0       # min time between flips of one patty
    jump_dist_frac: float = 0.5         # center jump (fraction of patty diameter) counted as displacement cue
    win_fill_frac: float = 0.45         # window must hold this share of expected samples
    require_disturbance: bool = True    # flip needs displacement/occlusion, not just a colour step

    # --- removal ---
    # a pro cook lifts the patty clear of the surface for several seconds during
    # a flip, so the "gone for good" timeout must sit well above that
    removed_after_s: float = 9.0        # lost longer than this (and not re-stitched) => removed
    eos_removed_after_s: float = 1.0    # at end of stream, accept this much absence as a removal

    # --- render ---
    draw_trails: bool = False
    font_scale: float = 0.55
