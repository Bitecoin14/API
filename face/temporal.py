from __future__ import annotations

import logging
import threading
from collections import Counter, deque
from dataclasses import dataclass, field

from .types import RecognitionStatus

log = logging.getLogger("hand_tracker.face.temporal")

_next_track_id = 0


def _new_track_id() -> int:
    global _next_track_id
    _next_track_id += 1
    return _next_track_id


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    decisions: deque = field(default_factory=deque)
    frames_since_seen: int = 0
    consecutive_same: int = 0
    initialized: bool = False


class TemporalSmoother:
    def __init__(
        self,
        window_size: int = 15,
        min_consensus: float = 0.60,
        init_frames: int = 3,
        timeout_frames: int = 30,
        iou_threshold: float = 0.4,
    ) -> None:
        self._window_size = window_size
        self._min_consensus = min_consensus
        self._init_frames = init_frames
        self._timeout_frames = timeout_frames
        self._iou_threshold = iou_threshold

        self._tracks: dict[int, Track] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        detections: list[tuple[tuple, str, RecognitionStatus]],
    ) -> dict[int, tuple[str, RecognitionStatus]]:
        with self._lock:
            # Step 1: age every existing track
            for track in self._tracks.values():
                track.frames_since_seen += 1

            # Step 2: match detections to tracks
            matched_track_ids: set[int] = set()

            for bbox, name, status in detections:
                best_tid: int | None = None
                best_iou = 0.0

                for tid, track in self._tracks.items():
                    if tid in matched_track_ids:
                        continue
                    score = _iou(bbox, track.bbox)
                    if score >= self._iou_threshold and score > best_iou:
                        best_iou = score
                        best_tid = tid

                if best_tid is not None:
                    # Update existing track
                    t = self._tracks[best_tid]
                    t.bbox = bbox
                    t.frames_since_seen = 0
                    matched_track_ids.add(best_tid)
                else:
                    # Create new track
                    new_tid = _new_track_id()
                    t = Track(
                        track_id=new_tid,
                        bbox=bbox,
                        decisions=deque(maxlen=self._window_size),
                    )
                    self._tracks[new_tid] = t
                    matched_track_ids.add(new_tid)
                    log.debug("New track %d created", new_tid)

                # Append decision to the matched/new track
                t.decisions.append((name, status))

                # Initialization check
                if not t.initialized:
                    recent = list(t.decisions)[-self._init_frames:]
                    if (
                        len(recent) >= self._init_frames
                        and len({n for n, _ in recent}) == 1
                    ):
                        t.initialized = True
                        log.debug("Track %d initialized on '%s'", t.track_id, name)

            # Step 3: remove stale tracks
            stale = [
                tid
                for tid, track in self._tracks.items()
                if track.frames_since_seen > self._timeout_frames
            ]
            for tid in stale:
                log.debug("Track %d timed out", tid)
                del self._tracks[tid]

            # Step 4: compute stable decision for each active track
            output: dict[int, tuple[str, RecognitionStatus]] = {}
            for tid, track in self._tracks.items():
                output[tid] = self._stable_decision(track)

            return output

    def get_track_id_for_bbox(self, bbox: tuple) -> int | None:
        with self._lock:
            best_tid: int | None = None
            best_iou = 0.0
            for tid, track in self._tracks.items():
                score = _iou(bbox, track.bbox)
                if score > best_iou:
                    best_iou = score
                    best_tid = tid
            return best_tid if best_iou > 0 else None

    def reset(self) -> None:
        with self._lock:
            self._tracks.clear()
            log.debug("TemporalSmoother reset")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _stable_decision(
        self, track: Track
    ) -> tuple[str, RecognitionStatus]:
        decisions = list(track.decisions)
        if not decisions:
            return ("?", RecognitionStatus.UNKNOWN)

        name_counts: Counter[str] = Counter(name for name, _ in decisions)
        most_common_name, top_count = name_counts.most_common(1)[0]
        fraction = top_count / len(decisions)

        if not track.initialized or fraction < self._min_consensus:
            return ("?", RecognitionStatus.UNKNOWN)

        # Most common status among the winning name's decisions only
        winning_statuses = [s for n, s in decisions if n == most_common_name]
        status_counts: Counter[RecognitionStatus] = Counter(winning_statuses)
        consensus_status = status_counts.most_common(1)[0][0]

        return (most_common_name, consensus_status)
