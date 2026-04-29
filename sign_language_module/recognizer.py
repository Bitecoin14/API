"""Rule-based ASL fingerspelling recognizer — letters A-Z.

Uses 42 landmarks: the 21 standard MediaPipe hand points plus 21 interpolated
midpoints along each bone connection. No external model required.
Classification is geometric: normalize → extension flags, key distances,
and hand orientation → decision tree.

Accuracy notes:
  High confidence: A B C D F I K L O S U V W Y
  Approximate:     E G H (need orientation), M N T (thumb-column geometry)
  Motion-based:    J ≈ I,  Z ≈ D  (static approximations)
  Orientation-dep: P ≈ K pointing down,  Q ≈ G pointing down
"""
from types import SimpleNamespace
from typing import Optional

import numpy as np

# ── Landmark indices (MediaPipe convention, 0-20) ──────────────────────────
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP     =  1,  2,  3,  4
INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP  =  5,  6,  7,  8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP =  9, 10, 11, 12
RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP   = 13, 14, 15, 16
PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP  = 17, 18, 19, 20

# ── Extended connections — 21 midpoints appended at indices 21-41 ──────────
_EXTENDED_CONNECTIONS = [
    (0,  1), (0,  5), (0, 17),
    (1,  2), (2,  3), (3,  4),
    (5,  6), (5,  9), (6,  7), (7,  8),
    (9, 10), (9, 13), (10, 11), (11, 12),
    (13, 14), (13, 17), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]

# ── Extended midpoint indices (21-41) ─────────────────────────────────────
MID_WRIST_THUMB_CMC = 21   # midpoint of (0,1)
MID_WRIST_INDEX_MCP = 22   # midpoint of (0,5)
MID_WRIST_PINKY_MCP = 23   # midpoint of (0,17)
MID_THUMB_CMC_MCP   = 24   # midpoint of (1,2)
MID_THUMB_MCP_IP    = 25   # midpoint of (2,3)
MID_THUMB_IP_TIP    = 26   # midpoint of (3,4)
MID_INDEX_MCP_PIP   = 27   # midpoint of (5,6)
MID_INDEX_MCP_MMCP  = 28   # midpoint of (5,9) — between index & middle MCPs
MID_INDEX_PIP_DIP   = 29   # midpoint of (6,7)
MID_INDEX_DIP_TIP   = 30   # midpoint of (7,8)
MID_MIDDLE_MCP_PIP  = 31   # midpoint of (9,10)
MID_MIDDLE_MCP_RMCP = 32   # midpoint of (9,13) — between middle & ring MCPs
MID_MIDDLE_PIP_DIP  = 33   # midpoint of (10,11)
MID_MIDDLE_DIP_TIP  = 34   # midpoint of (11,12)
MID_RING_MCP_PIP    = 35   # midpoint of (13,14)
MID_RING_MCP_PMCP   = 36   # midpoint of (13,17) — between ring & pinky MCPs
MID_RING_PIP_DIP    = 37   # midpoint of (14,15)
MID_RING_DIP_TIP    = 38   # midpoint of (15,16)
MID_PINKY_MCP_PIP   = 39   # midpoint of (17,18)
MID_PINKY_PIP_DIP   = 40   # midpoint of (18,19)
MID_PINKY_DIP_TIP   = 41   # midpoint of (19,20)


def compute_extended_landmarks(landmarks):
    """Extend 21 MediaPipe landmarks to 42 by inserting midpoints along each
    bone connection.  Returns a list of objects with x, y, z attributes.
    Indices 0-20: original MediaPipe landmarks.
    Indices 21-41: interpolated midpoints (order matches _EXTENDED_CONNECTIONS).
    """
    lm = landmarks
    extended = list(lm)
    for (i, j) in _EXTENDED_CONNECTIONS:
        a, b = lm[i], lm[j]
        extended.append(SimpleNamespace(
            x=(a.x + b.x) * 0.5,
            y=(a.y + b.y) * 0.5,
            z=(a.z + b.z) * 0.5,
        ))
    return extended


def _normalize(landmarks) -> np.ndarray:
    """Translate to wrist=origin, scale by wrist→middle-MCP distance."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[WRIST]
    scale = float(np.linalg.norm(pts[MIDDLE_MCP]))
    if scale > 1e-6:
        pts /= scale
    return pts  # shape (42, 3); y increases downward (image coords)


def _d(pts: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(pts[a] - pts[b]))


def _raised(pts: np.ndarray, tip: int, pip: int, margin: float = 0.07) -> bool:
    """Finger tip is above (smaller image-y) the PIP joint by at least margin."""
    return float(pts[tip][1]) < float(pts[pip][1]) - margin


def _fully_raised(pts: np.ndarray, tip: int, pip: int,
                  mid_dip_tip: int, margin: float = 0.07) -> bool:
    """Tip AND the DIP-TIP midpoint are both above PIP — confirms full extension."""
    return (_raised(pts, tip, pip, margin) and
            float(pts[mid_dip_tip][1]) < float(pts[pip][1]) + margin * 0.5)


def _between(a: float, b: float, val: float, margin: float = 0.18) -> bool:
    return min(a, b) - margin <= val <= max(a, b) + margin


def recognize_letter(hand_landmarks) -> Optional[str]:
    """Return the best-matching ASL letter or None when the posture is unclear."""
    ext = compute_extended_landmarks(hand_landmarks.landmark)
    pts = _normalize(ext)

    # ── Per-finger extension flags ─────────────────────────────────────────
    i_up = _raised(pts, INDEX_TIP,  INDEX_PIP)
    m_up = _raised(pts, MIDDLE_TIP, MIDDLE_PIP)
    r_up = _raised(pts, RING_TIP,   RING_PIP)
    p_up = _raised(pts, PINKY_TIP,  PINKY_PIP)

    # Thumb "out": tip is well away from the index-MCP base of the palm
    thumb_out = _d(pts, THUMB_TIP, INDEX_MCP) > 1.0

    # ── Key inter-landmark distances ───────────────────────────────────────
    d_ti = _d(pts, THUMB_TIP, INDEX_TIP)    # thumb–index pinch gap
    d_tm = _d(pts, THUMB_TIP, MIDDLE_TIP)   # thumb–middle tip gap
    d_im = _d(pts, INDEX_TIP, MIDDLE_TIP)   # index–middle tip spread

    # Extended-point distances for higher-accuracy measurements
    d_im_mid       = _d(pts, MID_INDEX_DIP_TIP, MID_MIDDLE_DIP_TIP)  # spread near fingertips
    d_thumb_k_arc  = _d(pts, MID_THUMB_IP_TIP, MIDDLE_DIP)           # thumb arc → middle DIP (K)

    # ── Hand orientation ───────────────────────────────────────────────────
    hand_down = float(pts[MIDDLE_MCP][1]) > 0.35

    idx_ax = abs(float(pts[INDEX_TIP][0]))
    idx_ay = abs(float(pts[INDEX_TIP][1]))
    index_sideways = idx_ax > idx_ay * 0.85

    mid_ax = abs(float(pts[MIDDLE_TIP][0]))
    mid_ay = abs(float(pts[MIDDLE_TIP][1]))
    both_sideways = index_sideways and (mid_ax > mid_ay * 0.85)

    # ── Classification ────────────────────────────────────────────────────

    # B — all four fingers up, thumb tucked
    if i_up and m_up and r_up and p_up:
        return 'B'

    # W — index + middle + ring up
    if i_up and m_up and r_up and not p_up:
        return 'W'

    # F — middle + ring + pinky up, thumb–index pinch (OK-like)
    if not i_up and m_up and r_up and p_up:
        return 'F' if d_ti < 0.65 else None

    # ── index + middle up ─────────────────────────────────────────────────
    if i_up and m_up and not r_up and not p_up:
        # K / P — thumb toward middle finger
        # MID_THUMB_IP_TIP proximity to MIDDLE_DIP adds a more reliable arc check
        if thumb_out and (d_tm < 1.2 or d_thumb_k_arc < 0.9):
            return 'P' if hand_down else 'K'
        # R — fingers crossed (tips very close together)
        if d_im < 0.28:
            return 'R'
        # H — both fingers pointing sideways
        if both_sideways:
            return 'H'
        # V vs U: DIP-TIP midpoint spread is more stable than raw tip distance
        return 'V' if (d_im_mid > 0.38 or d_im > 0.5) else 'U'

    # ── index only up ─────────────────────────────────────────────────────
    if i_up and not m_up and not r_up and not p_up:
        if index_sideways and thumb_out:
            return 'Q' if hand_down else 'G'
        if thumb_out:
            return 'L'
        return 'D'

    # ── pinky only up (± thumb) ───────────────────────────────────────────
    if not i_up and not m_up and not r_up and p_up:
        return 'Y' if thumb_out else 'I'

    # ── All fingers curled (fist-like) ───────────────────────────────────
    if not i_up and not m_up and not r_up and not p_up:

        # X — index is hooked: PIP raised but TIP curled back down
        pip_y    = float(pts[INDEX_PIP][1])
        mcp_y    = float(pts[INDEX_MCP][1])
        tip_y    = float(pts[INDEX_TIP][1])
        mid_mp_y = float(pts[MID_INDEX_MCP_PIP][1])  # confirms PIP section is raised
        if (pip_y < mcp_y - 0.15 and tip_y >= pip_y - 0.05 and
                not thumb_out and mid_mp_y < mcp_y - 0.08):
            return 'X'

        # O — thumb tip meets index tip forming a round loop
        if d_ti < 0.55:
            return 'O'

        # C — curved open hand: fingers arc outward, not tightly fisted
        index_reach     = _d(pts, INDEX_TIP, WRIST)
        index_mid_reach = _d(pts, MID_INDEX_DIP_TIP, WRIST)  # arc continuity check
        if d_ti > 1.15 and index_reach > 0.9 and index_mid_reach > 0.72:
            return 'C'

        # A — fist with thumb resting alongside
        if thumb_out:
            return 'A'

        # Distinguish S / T / N / M / E by where the thumb tip sits
        tx  = float(pts[THUMB_TIP][0])
        tx2 = float(pts[MID_THUMB_IP_TIP][0])  # secondary thumb position evidence
        ix  = float(pts[INDEX_MCP][0])
        mx  = float(pts[MIDDLE_MCP][0])
        rx  = float(pts[RING_MCP][0])
        px  = float(pts[PINKY_MCP][0])

        # T — thumb inserted between index and middle columns
        if _between(ix, mx, tx) or _between(ix, mx, tx2, margin=0.12):
            return 'T'

        # N — thumb under index+middle (sits in the middle–ring column gap)
        if _between(mx, rx, tx):
            return 'N'

        # M — thumb under three fingers (sits in the ring–pinky column gap)
        if _between(rx, px, tx):
            return 'M'

        # S — thumb wraps over the front of the fist (high up = low y value)
        if float(pts[THUMB_TIP][1]) < float(pts[INDEX_MCP][1]) - 0.12:
            return 'S'

        # E — fingers bent inward; DIP-TIP midpoints confirm the curl pattern
        e_curl = (float(pts[MID_INDEX_DIP_TIP][1])  > float(pts[INDEX_PIP][1])  - 0.10 and
                  float(pts[MID_MIDDLE_DIP_TIP][1]) > float(pts[MIDDLE_PIP][1]) - 0.10)
        return 'E' if e_curl else None

    return None
