import cv2
import mediapipe as mp
import numpy as np

_mp_pose = mp.solutions.pose

# Connections grouped by body region for color-coding
_FACE_CONN = {(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10)}
_TORSO_CONN = {(11,12),(11,23),(12,24),(23,24)}
_LEFT_ARM_CONN = {(11,13),(13,15),(15,17),(15,19),(15,21),(17,19)}
_RIGHT_ARM_CONN = {(12,14),(14,16),(16,18),(16,20),(16,22),(18,20)}
_LEFT_LEG_CONN = {(23,25),(25,27),(27,29),(27,31),(29,31)}
_RIGHT_LEG_CONN = {(24,26),(26,28),(28,30),(28,32),(30,32)}

_CONN_COLORS = [
    (_FACE_CONN,      (200, 200, 200)),   # gray
    (_TORSO_CONN,     (0, 255, 255)),     # cyan
    (_LEFT_ARM_CONN,  (0, 255, 0)),       # green
    (_RIGHT_ARM_CONN, (255, 80, 0)),      # blue-orange
    (_LEFT_LEG_CONN,  (0, 165, 255)),     # orange
    (_RIGHT_LEG_CONN, (80, 80, 255)),     # red
]

_VIS_THRESHOLD = 0.5
_BONE_THICKNESS = 2
_JOINT_RADIUS = 5


class PoseDetector:
    def __init__(self, min_detection_confidence=0.6, min_tracking_confidence=0.6,
                 model_complexity=1):
        self._pose = _mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
            enable_segmentation=False,
        )

    def process(self, rgb_frame: np.ndarray):
        return self._pose.process(rgb_frame)

    def draw(self, frame: np.ndarray, pose_results, coord_transform=None) -> np.ndarray:
        if pose_results is None or not pose_results.pose_landmarks:
            return frame

        h, w = frame.shape[:2]
        lm = pose_results.pose_landmarks.landmark

        pts = {}
        for idx, landmark in enumerate(lm):
            x = int(round(landmark.x * w))
            y = int(round(landmark.y * h))
            if coord_transform is not None:
                x, y = coord_transform(x, y, w, h)
            pts[idx] = (x, y)

        for conn_set, color in _CONN_COLORS:
            for start, end in conn_set:
                if lm[start].visibility > _VIS_THRESHOLD and lm[end].visibility > _VIS_THRESHOLD:
                    cv2.line(frame, pts[start], pts[end], color, _BONE_THICKNESS, cv2.LINE_AA)

        for idx, pt in pts.items():
            if lm[idx].visibility > _VIS_THRESHOLD:
                cv2.circle(frame, pt, _JOINT_RADIUS, (255, 255, 255), cv2.FILLED)
                cv2.circle(frame, pt, _JOINT_RADIUS, (0, 0, 0), 1, cv2.LINE_AA)

        return frame

    def close(self) -> None:
        self._pose.close()
