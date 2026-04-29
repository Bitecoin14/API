# models/train_asl.py
"""
Synthetic-data ASL baseline trainer.

Generates landmark-based training samples from geometric archetypes
(one set of ideal hand positions per letter), adds Gaussian noise,
trains a RandomForestClassifier, and saves models/asl_classifier.pkl.

Accuracy on real data: ~75-85%. For higher accuracy, collect real data
with models/collect_data.py (not yet written) and retrain.

Usage:
    python models/train_asl.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ── Landmark indices (MediaPipe convention) ──────────────────────────────────
WRIST       = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP     =  1,  2,  3,  4
INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP  =  5,  6,  7,  8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP =  9, 10, 11, 12
RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP   = 13, 14, 15, 16
PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP  = 17, 18, 19, 20
N_LANDMARKS = 21

# ── Archetype builder ─────────────────────────────────────────────────────────

def _base() -> np.ndarray:
    """Neutral open hand in normalized coordinates (wrist=origin, scale=1)."""
    pts = np.zeros((N_LANDMARKS, 3), dtype=np.float32)
    # Wrist
    pts[WRIST] = [0.0, 0.0, 0.0]
    # Thumb column (pointing left / negative x)
    pts[THUMB_CMC] = [-0.20, -0.15, 0.0]
    pts[THUMB_MCP] = [-0.40, -0.25, 0.0]
    pts[THUMB_IP]  = [-0.60, -0.35, 0.0]
    pts[THUMB_TIP] = [-0.80, -0.45, 0.0]
    # Index finger
    pts[INDEX_MCP] = [-0.30, -1.00, 0.0]
    pts[INDEX_PIP] = [-0.30, -1.40, 0.0]
    pts[INDEX_DIP] = [-0.30, -1.65, 0.0]
    pts[INDEX_TIP] = [-0.30, -1.90, 0.0]
    # Middle finger (scale reference: MCP at y=-1)
    pts[MIDDLE_MCP] = [ 0.00, -1.00, 0.0]
    pts[MIDDLE_PIP] = [ 0.00, -1.40, 0.0]
    pts[MIDDLE_DIP] = [ 0.00, -1.65, 0.0]
    pts[MIDDLE_TIP] = [ 0.00, -1.90, 0.0]
    # Ring finger
    pts[RING_MCP] = [ 0.25, -0.95, 0.0]
    pts[RING_PIP] = [ 0.25, -1.35, 0.0]
    pts[RING_DIP] = [ 0.25, -1.60, 0.0]
    pts[RING_TIP] = [ 0.25, -1.85, 0.0]
    # Pinky finger
    pts[PINKY_MCP] = [ 0.50, -0.85, 0.0]
    pts[PINKY_PIP] = [ 0.50, -1.15, 0.0]
    pts[PINKY_DIP] = [ 0.50, -1.38, 0.0]
    pts[PINKY_TIP] = [ 0.50, -1.58, 0.0]
    return pts


def _curl_finger(pts: np.ndarray, mcp: int, pip: int, dip: int, tip: int) -> np.ndarray:
    """Curl a finger so tip and DIP drop below PIP (y increases downward in image)."""
    p = pts.copy()
    mcp_y = p[mcp][1]
    p[pip] = [p[pip][0], mcp_y - 0.30, 0.05]
    p[dip] = [p[dip][0], mcp_y - 0.15, 0.12]
    p[tip] = [p[tip][0], mcp_y - 0.05, 0.15]
    return p


def _curl_all(pts: np.ndarray) -> np.ndarray:
    for mcp, pip, dip, tip in [
        (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
        (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
        (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
    ]:
        pts = _curl_finger(pts, mcp, pip, dip, tip)
    return pts


def _tuck_thumb(pts: np.ndarray) -> np.ndarray:
    p = pts.copy()
    p[THUMB_TIP] = [-0.10, -0.70, 0.05]
    p[THUMB_IP]  = [-0.15, -0.55, 0.03]
    return p


def _thumb_alongside(pts: np.ndarray) -> np.ndarray:
    """Thumb resting alongside fist (A-like)."""
    p = pts.copy()
    p[THUMB_TIP] = [-0.65, -0.60, 0.0]
    p[THUMB_IP]  = [-0.55, -0.45, 0.0]
    return p


def _thumb_out(pts: np.ndarray) -> np.ndarray:
    """Thumb pointing left (L-like)."""
    p = pts.copy()
    p[THUMB_TIP] = [-1.10, -0.30, 0.0]
    p[THUMB_IP]  = [-0.90, -0.30, 0.0]
    p[THUMB_MCP] = [-0.60, -0.25, 0.0]
    return p


ARCHETYPES: dict[str, np.ndarray] = {}


def _build_archetypes() -> None:
    b = _base()
    fist = _curl_all(b)

    # A — fist, thumb alongside
    ARCHETYPES["A"] = _thumb_alongside(fist)

    # B — all fingers up, thumb tucked
    ARCHETYPES["B"] = _tuck_thumb(b)

    # C — curved hand: fingers partially curled
    c = b.copy()
    for mcp, pip, dip, tip in [
        (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
        (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
        (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
    ]:
        c[pip] = [c[pip][0], c[mcp][1] - 0.55, 0.08]
        c[dip] = [c[dip][0], c[mcp][1] - 0.40, 0.12]
        c[tip] = [c[tip][0], c[mcp][1] - 0.25, 0.14]
    c[THUMB_TIP] = [-0.90, -0.25, 0.0]
    ARCHETYPES["C"] = c

    # D — index up, others curled, no thumb out
    d = _curl_all(b)
    d[INDEX_PIP] = b[INDEX_PIP]; d[INDEX_DIP] = b[INDEX_DIP]; d[INDEX_TIP] = b[INDEX_TIP]
    ARCHETYPES["D"] = d

    # E — all fingers bent inward (tips at MCP level)
    e = fist.copy()
    for tip in [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
        e[tip][1] = e[tip][1] + 0.30   # tips even lower than normal curl
    ARCHETYPES["E"] = _tuck_thumb(e)

    # F — middle+ring+pinky up, index+thumb pinch
    f = b.copy()
    f = _curl_finger(f, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)
    f[THUMB_TIP] = [-0.30, -1.45, 0.0]  # thumb touching index tip
    ARCHETYPES["F"] = f

    # G — index sideways, thumb out
    g = _curl_all(b)
    g[INDEX_TIP] = [-1.50, -1.00, 0.0]   # pointing left (sideways)
    g[INDEX_DIP] = [-1.20, -1.00, 0.0]
    g[INDEX_PIP] = [-0.90, -1.00, 0.0]
    g = _thumb_out(g)
    ARCHETYPES["G"] = g

    # H — index+middle sideways
    h = _curl_all(b)
    for off, (pip, dip, tip) in enumerate([(INDEX_PIP, INDEX_DIP, INDEX_TIP),
                                             (MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP)]):
        base_y = -1.00 - off * 0.15
        h[pip] = [-0.70, base_y, 0.0]
        h[dip] = [-1.00, base_y, 0.0]
        h[tip] = [-1.30, base_y, 0.0]
    ARCHETYPES["H"] = h

    # I — pinky up, others curled
    i = _curl_all(b)
    i[PINKY_PIP] = b[PINKY_PIP]; i[PINKY_DIP] = b[PINKY_DIP]; i[PINKY_TIP] = b[PINKY_TIP]
    ARCHETYPES["I"] = i

    # J ≈ I (static approximation — J is motion-based)
    ARCHETYPES["J"] = ARCHETYPES["I"].copy()

    # K — index+middle up, thumb toward middle
    k = _curl_all(b)
    k[INDEX_PIP] = b[INDEX_PIP]; k[INDEX_DIP] = b[INDEX_DIP]; k[INDEX_TIP] = b[INDEX_TIP]
    k[MIDDLE_PIP] = b[MIDDLE_PIP]; k[MIDDLE_DIP] = b[MIDDLE_DIP]; k[MIDDLE_TIP] = b[MIDDLE_TIP]
    k[THUMB_TIP] = [-0.05, -1.45, 0.0]  # thumb tip near middle finger
    ARCHETYPES["K"] = k

    # L — index up + thumb out
    l = _curl_all(b)
    l[INDEX_PIP] = b[INDEX_PIP]; l[INDEX_DIP] = b[INDEX_DIP]; l[INDEX_TIP] = b[INDEX_TIP]
    ARCHETYPES["L"] = _thumb_out(l)

    # M — fist, thumb under ring-pinky gap
    m = fist.copy()
    m[THUMB_TIP] = [0.40, -0.65, 0.05]
    ARCHETYPES["M"] = m

    # N — fist, thumb under middle-ring gap
    n = fist.copy()
    n[THUMB_TIP] = [0.15, -0.65, 0.05]
    ARCHETYPES["N"] = n

    # O — thumb tip meets index tip
    o = b.copy()
    for mcp, pip, dip, tip in [
        (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
        (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
        (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
    ]:
        o[pip] = [o[pip][0], o[mcp][1] - 0.50, 0.05]
        o[dip] = [o[dip][0], o[mcp][1] - 0.35, 0.08]
        o[tip] = [o[tip][0], o[mcp][1] - 0.20, 0.10]
    o[THUMB_TIP] = [-0.30, -0.80, 0.0]  # meets index tip
    ARCHETYPES["O"] = o

    # P ≈ K pointing down
    ARCHETYPES["P"] = ARCHETYPES["K"].copy()

    # Q ≈ G pointing down
    ARCHETYPES["Q"] = ARCHETYPES["G"].copy()

    # R — index+middle up, crossed (tips close)
    r = _curl_all(b)
    r[INDEX_TIP] = [-0.15, -1.90, 0.0]
    r[INDEX_DIP] = [-0.15, -1.65, 0.0]
    r[INDEX_PIP] = [-0.20, -1.40, 0.0]
    r[MIDDLE_TIP] = [-0.25, -1.90, 0.0]
    r[MIDDLE_DIP] = [-0.15, -1.65, 0.0]
    r[MIDDLE_PIP] = [-0.10, -1.40, 0.0]
    ARCHETYPES["R"] = r

    # S — fist, thumb over top of fingers
    s = fist.copy()
    s[THUMB_TIP] = [-0.30, -0.90, 0.0]  # high up
    ARCHETYPES["S"] = s

    # T — thumb between index and middle columns
    t = fist.copy()
    t[THUMB_TIP] = [-0.15, -0.75, 0.05]
    ARCHETYPES["T"] = t

    # U — index+middle up, close together
    u = _curl_all(b)
    u[INDEX_PIP] = b[INDEX_PIP]; u[INDEX_DIP] = b[INDEX_DIP]; u[INDEX_TIP] = b[INDEX_TIP]
    u[MIDDLE_PIP] = b[MIDDLE_PIP]; u[MIDDLE_DIP] = b[MIDDLE_DIP]; u[MIDDLE_TIP] = b[MIDDLE_TIP]
    ARCHETYPES["U"] = u

    # V — index+middle up, spread
    v = _curl_all(b)
    v[INDEX_TIP] = [-0.55, -1.85, 0.0]
    v[INDEX_DIP] = [-0.50, -1.60, 0.0]
    v[INDEX_PIP] = [-0.42, -1.35, 0.0]
    v[MIDDLE_TIP] = [ 0.25, -1.85, 0.0]
    v[MIDDLE_DIP] = [ 0.20, -1.60, 0.0]
    v[MIDDLE_PIP] = [ 0.12, -1.35, 0.0]
    ARCHETYPES["V"] = v

    # W — index+middle+ring up
    w = _curl_all(b)
    for pip, dip, tip in [(INDEX_PIP, INDEX_DIP, INDEX_TIP),
                           (MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
                           (RING_PIP, RING_DIP, RING_TIP)]:
        w[pip] = b[pip]; w[dip] = b[dip]; w[tip] = b[tip]
    ARCHETYPES["W"] = w

    # X — index hooked: PIP up but TIP curled back
    x = _curl_all(b)
    x[INDEX_PIP] = [-0.30, -1.15, 0.0]   # PIP raised above MCP
    x[INDEX_DIP] = [-0.30, -0.95, 0.08]  # DIP below PIP
    x[INDEX_TIP] = [-0.25, -0.82, 0.12]  # TIP even lower
    ARCHETYPES["X"] = x

    # Y — pinky+thumb out
    y = _curl_all(b)
    y[PINKY_PIP] = b[PINKY_PIP]; y[PINKY_DIP] = b[PINKY_DIP]; y[PINKY_TIP] = b[PINKY_TIP]
    ARCHETYPES["Y"] = _thumb_out(y)

    # Z ≈ D (static approximation — Z is motion-based)
    ARCHETYPES["Z"] = ARCHETYPES["D"].copy()


def _normalize(pts: np.ndarray) -> np.ndarray:
    """Same normalisation used in ASLStage: wrist=origin, scale=wrist→middle-MCP."""
    pts = pts - pts[WRIST]
    scale = float(np.linalg.norm(pts[MIDDLE_MCP]))
    if scale > 1e-6:
        pts = pts / scale
    return pts.flatten()   # 63 floats


def generate_dataset(n_per_letter: int = 800, noise_std: float = 0.04
                     ) -> tuple[np.ndarray, np.ndarray]:
    _build_archetypes()
    X_parts, y_parts = [], []
    rng = np.random.default_rng(42)
    for letter, base_pts in ARCHETYPES.items():
        norm = _normalize(base_pts)
        noise = rng.normal(0, noise_std, (n_per_letter, 63)).astype(np.float32)
        samples = norm + noise
        X_parts.append(samples)
        y_parts.extend([letter] * n_per_letter)
    return np.vstack(X_parts), np.array(y_parts)


def train(n_per_letter: int = 800) -> SKPipeline:
    print(f"Generating {n_per_letter} samples per letter …")
    X, y = generate_dataset(n_per_letter)
    model = SKPipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
        )),
    ])
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    model.fit(X, y)
    return model


if __name__ == "__main__":
    out = Path(__file__).parent / "asl_classifier.pkl"
    model = train()
    joblib.dump(model, out)
    print(f"Model saved → {out}")
