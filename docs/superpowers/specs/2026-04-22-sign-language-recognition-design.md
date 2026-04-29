# Sign Language Recognition — Design Spec
**Date:** 2026-04-22  
**Status:** Approved

---

## Problem

The existing recognizer uses only a subset of the 42 available landmarks. Several letter pairs share nearly identical feature sets and are resolved by a single fragile threshold. No unit tests exist for the recognition logic, so regressions are invisible and threshold tuning is guesswork.

**Known confusion pairs:** M/N/T (overlapping thumb column), S/E/A (all-fingers-curled variants), K/P (orientation only), G/Q (orientation only), V/U (single spread threshold).

---

## Goals

1. Use all 42 landmarks (21 MediaPipe + 21 midpoints) in the recognizer.
2. Resolve all known confusion pairs with additional geometric features.
3. Add tertile interpolation (42 → 72 points) only if discriminability tests show separation < 0.15.
4. Provide a full unit test suite with synthetic hand data covering all 26 letters and all confusion pairs.

---

## Architecture

Two files change; everything else is untouched.

### `sign_language_module/recognizer.py`

A new function `_extract_features(pts)` sits between normalization and the decision tree. It consumes all 42 normalized points and returns a flat dict of ~35 named geometric values. The decision tree reads exclusively from this dict — no geometry is computed inline.

Every threshold that gates a decision becomes a named constant at the top of the file, annotated with which letters it affects. This makes tuning a threshold a one-line change that does not require reading the tree logic.

`compute_extended_landmarks` may grow from 42 → 72 points (adding ⅓ and ⅔ tertile positions along the 5 finger chains: 5 fingers × 3 segments × 2 points = 30 extra). This is added **only if** one or more discriminability tests fail.

### `test_sign_language_recognition.py` (new)

Synthetic test file. No camera, no MediaPipe runtime required. Contains a `_make_pose` helper and four test classes described in the Testing section.

---

## Feature Extraction

`_extract_features(pts: np.ndarray) -> dict`

Input: normalized 42-point array (WRIST at origin, scale = wrist→MIDDLE_MCP distance).  
Output: flat dict with the following groups.

### Extension & Curl (per finger: thumb, index, middle, ring, pinky)

| Key | Type | Description |
|-----|------|-------------|
| `raised[f]` | bool | tip above PIP by `MARGIN_RAISED` |
| `fully_raised[f]` | bool | tip AND DIP-TIP midpoint both above PIP |
| `curl[f]` | float 0–1 | how far tip has dropped back toward palm relative to PIP; uses DIP-TIP midpoint for a two-sample average |

Curl formula for each finger:
```
pip_y   = pts[PIP][1]
tip_y   = pts[TIP][1]
mid_y   = pts[MID_DIP_TIP][1]
avg_y   = (tip_y + mid_y) / 2
curl[f] = clamp((avg_y - pip_y) / CURL_SCALE, 0.0, 1.0)
```
`CURL_SCALE` is the expected vertical drop for a fully curled finger (~0.6 in normalized units).

### Multi-Level Spread

Spread measured at three heights for each adjacent finger pair (index-middle, middle-ring, ring-pinky):

| Key | Description |
|-----|-------------|
| `spread_mcp[pair]` | distance between adjacent MCPs |
| `spread_pip[pair]` | distance between adjacent PIP-level midpoints |
| `spread_tip[pair]` | distance between adjacent DIP-TIP midpoints |

This replaces the single `d_im` value with a three-reading profile. V has progressive widening toward the tip; U has uniform narrow spread throughout all three levels.

### Thumb Geometry

| Key | Type | Description |
|-----|------|-------------|
| `thumb_out` | bool | `d(THUMB_TIP, INDEX_MCP) > THUMB_OUT_RADIUS` |
| `thumb_tip_x` | float | normalized x of THUMB_TIP (column position) |
| `thumb_tip_y` | float | normalized y of THUMB_TIP (row position) |
| `thumb_arc_x` | float | normalized x of MID_THUMB_IP_TIP |
| `thumb_arc_y` | float | normalized y of MID_THUMB_IP_TIP |
| `d_ti` | float | THUMB_TIP ↔ INDEX_TIP |
| `d_tm` | float | THUMB_TIP ↔ MIDDLE_TIP |
| `d_thumb_k_arc` | float | MID_THUMB_IP_TIP ↔ MIDDLE_DIP (K/P arc) |
| `thumb_over_fist` | bool | `THUMB_TIP.y < INDEX_MCP.y − THUMB_OVER_MARGIN` (S) |
| `thumb_tip_z` | float | normalized z of THUMB_TIP (forward depth, for M/N/T) |

### Inter-Finger Distances

| Key | Description |
|-----|-------------|
| `d_im` | INDEX_TIP ↔ MIDDLE_TIP (V/U, R) |
| `d_im_mid` | MID_INDEX_DIP_TIP ↔ MID_MIDDLE_DIP_TIP (V/U) |
| `d_index_wrist` | INDEX_TIP ↔ WRIST (C reach) |
| `d_index_mid_wrist` | MID_INDEX_DIP_TIP ↔ WRIST (C arc continuity) |

### Hand Orientation

| Key | Type | Description |
|-----|------|-------------|
| `hand_down` | bool | two-point average of MIDDLE_MCP.y and MID_WRIST_INDEX_MCP.y > `HAND_DOWN_THRESH` — more robust to wrist roll than single-point check |
| `index_sideways` | bool | `\|INDEX_TIP.x\| > \|INDEX_TIP.y\| × SIDEWAYS_RATIO` |
| `both_sideways` | bool | index_sideways AND `\|MIDDLE_TIP.x\| > \|MIDDLE_TIP.y\| × SIDEWAYS_RATIO` |

### Index Hook Features (X)

| Key | Description |
|-----|-------------|
| `index_pip_y` | normalized y of INDEX_PIP |
| `index_mcp_y` | normalized y of INDEX_MCP |
| `index_tip_y` | normalized y of INDEX_TIP |
| `mid_index_mcp_pip_y` | normalized y of MID_INDEX_MCP_PIP |

### MCP Column Positions (M/N/T/S)

| Key | Description |
|-----|-------------|
| `mcp_x[f]` | normalized x of each finger's MCP (index, middle, ring, pinky) |

---

## Named Threshold Constants

All thresholds live at the top of `recognizer.py`, grouped by the letters they affect:

```python
# Extension / curl
MARGIN_RAISED      = 0.07   # raised[f]: A-Z all uses
CURL_SCALE         = 0.60   # curl[f] denominator
CURL_E_MIN         = 0.35   # E: minimum curl for all four fingers

# Thumb
THUMB_OUT_RADIUS   = 1.00   # thumb_out: A, L, Y, G, Q, K, P
THUMB_OVER_MARGIN  = 0.12   # thumb_over_fist: S

# F
D_TI_PINCH         = 0.65   # F: thumb-index pinch

# K / P
D_TM_K             = 1.20   # K/P: thumb-middle distance
D_THUMB_ARC_K      = 0.90   # K/P: MID_THUMB_IP_TIP ↔ MIDDLE_DIP

# R
D_IM_CROSSED       = 0.28   # R: crossed fingers tip distance

# V / U
SPREAD_V_MID       = 0.38   # V: DIP-TIP midpoint spread
D_IM_V             = 0.50   # V: tip-to-tip spread (backup)

# O
D_TI_O             = 0.55   # O: thumb-index pinch loop

# C
D_TI_C             = 1.15   # C: wide thumb-index gap
REACH_C            = 0.90   # C: index tip reach from wrist
REACH_C_MID        = 0.72   # C: DIP-TIP midpoint reach from wrist

# X
PIP_HOOK           = 0.15   # X: PIP raised above MCP
TIP_HOOK           = 0.05   # X: tip not above PIP
MID_HOOK           = 0.08   # X: MID_INDEX_MCP_PIP elevation

# M / N / T (column margins)
BETWEEN_MARGIN     = 0.18   # T/N/M primary column check
BETWEEN_MARGIN_T2  = 0.12   # T secondary arc check

# Orientation
HAND_DOWN_THRESH   = 0.35   # hand_down
SIDEWAYS_RATIO     = 0.85   # index_sideways / both_sideways
```

---

## Decision Tree

The tree structure is unchanged. Each branch reads from the feature dict `f`:

```
1. All four raised                      → B
2. index + middle + ring raised         → W
3. middle + ring + pinky raised
     d_ti < D_TI_PINCH                 → F  else None
4. index + middle raised
     thumb_out AND (d_tm < D_TM_K
       OR d_thumb_k_arc < D_THUMB_ARC_K)
         hand_down                     → P  else K
     d_im < D_IM_CROSSED               → R
     both_sideways                     → H
     spread_tip[im] > SPREAD_V_MID
       OR d_im > D_IM_V                → V  else U
5. index only raised
     index_sideways AND thumb_out
         hand_down                     → Q  else G
     thumb_out                         → L
     else                              → D
6. pinky only raised
     thumb_out                         → Y  else I
7. All curled
     hook check                        → X
     d_ti < D_TI_O                    → O
     d_ti > D_TI_C AND reach checks   → C
     thumb_out                         → A
     between(ix, mx, tx OR tx2)        → T
     between(mx, rx, tx)               → N
     between(rx, px, tx)               → M
     thumb_over_fist                   → S
     all curl[f] > CURL_E_MIN          → E  else None
8. else                                → None
```

### Confusion Pair Fixes

**M / N / T** — The `_between` column check is supplemented with `thumb_tip_z`: T has the thumb pushed most forward (toward camera, negative z in normalized space), since it sits under just one finger. M has the thumb pushed least forward (three fingers over it). This 3-D cue breaks ties at column boundaries. Additionally, the count of fingers whose DIP-TIP midpoints are vertically above the thumb tip is computed: T = 1 finger, N = 2 fingers, M = 3 fingers.

**S / E** — S requires `thumb_over_fist` (thumb tip above the knuckle line). E requires `all curl[f] > CURL_E_MIN` for all four fingers using the DIP-TIP midpoint curl metric. Both conditions are mutually exclusive: a thumb that is over the fist can't simultaneously have the DIP-TIP midpoints below PIP level.

**hand_down (K/P, G/Q)** — Computed as the average of `MIDDLE_MCP.y` and `MID_WRIST_INDEX_MCP.y` rather than MIDDLE_MCP alone, reducing sensitivity to wrist roll.

**V / U** — Uses spread at all three levels (MCP, PIP, DIP-TIP). V shows a widening profile (spread_tip > spread_mcp); U shows a uniform narrow profile. A letter is returned as V only if the tip-level spread exceeds `SPREAD_V_MID` OR the raw tip distance exceeds `D_IM_V`.

---

## Tertile Interpolation (conditional)

If the discriminability tests report separation < 0.15 for any confusion pair, `compute_extended_landmarks` is extended to add ⅓ and ⅔ positions along the 5 finger chains:

```
For each finger (thumb, index, middle, ring, pinky):
  For each segment (CMC→MCP, MCP→PIP, PIP→DIP, DIP→TIP):
    Add point at t=1/3 and t=2/3
```

This gives 5 × 3 × 2 = 30 additional points. These do not overlap with the existing 21 midpoints (which sit at t=½); the tertile points sit at t=⅓ and t=⅔. Total landmarks: 42 + 30 = 72. The most likely trigger is M/N/T; the tertile points along the proximal phalanges provide a cleaner measure of which fingers are folded over the thumb in 3-D.

---

## Testing

### Reference Frame for Synthetic Poses

All synthetic poses use WRIST = (0, 0, 0) and MIDDLE_MCP = (0, −1, 0). Because `_normalize` subtracts WRIST (→ no change) and divides by ‖MIDDLE_MCP‖ = 1.0 (→ no change), the coordinates fed in equal the coordinates the recognizer sees. Midpoints computed by `compute_extended_landmarks` are therefore also in the expected normalized positions.

### `_make_pose` Helper

Accepts named parameters:
- `fingers_up`: list from `['thumb', 'index', 'middle', 'ring', 'pinky']`
- `thumb`: `'out' | 'tucked' | 'pinch' | 'over_fist' | 'between_cols'`
- `spread`: float 0–1 (finger spread at all levels)
- `hand_down`: bool
- `index_sideways`: bool
- `thumb_col`: int 0–3 (column position for T/N/M, overrides thumb x)

Returns `FakeHandLandmarks` with a `.landmark` list of 21 `SimpleNamespace(x, y, z)` objects. Each named finger/thumb state maps to a fixed set of verified (x, y, z) coordinates that the test suite itself validates in `TestFeatureExtraction`.

### `TestFeatureExtraction`

Unit tests for `_extract_features` in isolation:
- `raised['index']` is True for an extended index finger (tip.y = −2.15), False for curled (tip.y = −0.9)
- `curl['index']` is ≤ 0.05 for straight, ≥ 0.90 for fully folded
- `thumb_out` is True when tip is 1.22 units from INDEX_MCP, False when 0.2 units
- `hand_down` is True when palm faces downward, False when upward
- `spread_tip['im']` widens correctly as `spread` parameter increases

### `TestLetterRecognition`

26 tests, one per letter. Each calls `recognize_letter(pose)` on a canonical synthetic pose and asserts the exact returned string. J asserts `'I'`; Z asserts `'D'` (static approximations, documented). All 26 must pass before implementation is considered complete.

### `TestDiscriminability`

One test per confusion pair: M/N, N/T, T/M, S/E, S/A, E/A, K/P, G/Q, V/U.

Each test:
1. Builds both canonical poses.
2. Calls `_extract_features` on both (bypassing `recognize_letter`).
3. For each feature that distinguishes the pair, asserts `|f_a[key] − f_b[key]| > MIN_SEPARATION` (0.15).
4. On failure, prints the feature name and the actual separation, making it actionable.

Failing a discriminability test triggers adding tertile interpolation for that pair's distinguishing dimension.

### `TestEdgeCases`

- Thumb at the exact T/N x-boundary: result is `'T'` or `'N'`, not a crash or silent `None`.
- Index finger at raise/curl threshold ± 0.02: result is consistent (no flip).
- Index finger at 45° between up and sideways: returns `None` rather than confidently picking a letter.
- Both hands present: each hand's result is independent.

---

## Files Changed

| File | Change |
|------|--------|
| `sign_language_module/recognizer.py` | Add `_extract_features`; refactor tree to use feature dict; add named threshold constants; optionally extend to 72 points |
| `test_sign_language_recognition.py` | New file — all four test classes |
| `sign_language_module/__init__.py` | No change needed |
| `hand_tracker.py` | No change needed |
