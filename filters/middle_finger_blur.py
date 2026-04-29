import cv2
import numpy as np

# MediaPipe landmark indices
_MIDDLE_TIP = 12
_MIDDLE_PIP = 10
_INDEX_TIP  = 8
_INDEX_PIP  = 6
_RING_TIP   = 16
_RING_PIP   = 14
_PINKY_TIP  = 20
_PINKY_PIP  = 18

_BLUR_KERNEL = (51, 51)
_PAD = 30  # padding around bounding box when blurring


def _is_middle_finger_raised(hand_landmarks) -> bool:
    lm = hand_landmarks.landmark
    middle_up = lm[_MIDDLE_TIP].y < lm[_MIDDLE_PIP].y
    index_down = lm[_INDEX_TIP].y > lm[_INDEX_PIP].y
    ring_down  = lm[_RING_TIP].y  > lm[_RING_PIP].y
    pinky_down = lm[_PINKY_TIP].y > lm[_PINKY_PIP].y
    return middle_up and index_down and ring_down and pinky_down


def _apply(frame: np.ndarray, hand_results=None, coord_transform=None, **kwargs) -> np.ndarray:
    if hand_results is None or not hand_results.multi_hand_landmarks:
        return frame

    h, w = frame.shape[:2]
    out  = frame.copy()

    for hand_landmarks in hand_results.multi_hand_landmarks:
        if not _is_middle_finger_raised(hand_landmarks):
            continue

        # Compute landmark positions, applying the active filter's coord transform
        # so the blur region lands on the correct spot in the (possibly flipped) frame.
        if coord_transform is not None:
            pts = [
                coord_transform(int(round(lm.x * w)), int(round(lm.y * h)), w, h)
                for lm in hand_landmarks.landmark
            ]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
        else:
            xs = [int(round(lm.x * w)) for lm in hand_landmarks.landmark]
            ys = [int(round(lm.y * h)) for lm in hand_landmarks.landmark]

        x1 = max(min(xs) - _PAD, 0)
        y1 = max(min(ys) - _PAD, 0)
        x2 = min(max(xs) + _PAD, w)
        y2 = min(max(ys) + _PAD, h)

        roi     = out[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(roi, _BLUR_KERNEL, 0)
        out[y1:y2, x1:x2] = blurred

    return out


FILTER = {"name": "Middle Finger Blur", "apply": _apply}
