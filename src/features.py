"""Cálculo de features geométricas a partir de keypoints COCO."""

import math
import numpy as np

from .constants import WRIST_HEIGHT_MIN, WRIST_HEIGHT_MAX


def _angle_3p(p1, p2, p3):
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(*v1); n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return np.nan
    cos = max(-1, min(1, dot / (n1 * n2)))
    return math.degrees(math.acos(cos))


def _angle_with_vertical(p_top, p_bottom):
    dx = p_top[0] - p_bottom[0]
    dy = p_top[1] - p_bottom[1]
    if dx == 0 and dy == 0:
        return np.nan
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def _distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def _safe_product(*values):
    if any(np.isnan(v) for v in values):
        return np.nan
    r = 1.0
    for v in values:
        r *= v
    return r


def compute_features(kp_xy, kp_conf):
    """Calcula las 13 features tabulares a partir de los keypoints COCO.

    Args:
        kp_xy: array (17, 2) con coordenadas (x, y) de cada keypoint.
        kp_conf: array (17,) con la confianza de cada keypoint.

    Returns:
        Diccionario con las 13 features, o None si la pose es degenerada
        (torso de longitud casi cero).
    """
    # Indices COCO
    LEFT_IDX = {"shoulder": 5, "hip": 11, "knee": 13, "ankle": 15, "wrist": 9, "ear": 3}
    RIGHT_IDX = {"shoulder": 6, "hip": 12, "knee": 14, "ankle": 16, "wrist": 10, "ear": 4}

    # Determinar lado más visible
    left_conf = np.mean([kp_conf[i] for i in (5, 11, 13, 15, 9)])
    right_conf = np.mean([kp_conf[i] for i in (6, 12, 14, 16, 10)])
    side_idx = RIGHT_IDX if right_conf >= left_conf else LEFT_IDX
    side_name = "right" if right_conf >= left_conf else "left"

    def kp(idx):
        return float(kp_xy[idx][0]), float(kp_xy[idx][1])

    shoulder = kp(side_idx["shoulder"])
    hip      = kp(side_idx["hip"])
    knee     = kp(side_idx["knee"])
    ankle    = kp(side_idx["ankle"])
    ear      = kp(side_idx["ear"])

    l_shoulder = kp(5); r_shoulder = kp(6)
    l_wrist    = kp(9); r_wrist    = kp(10)
    l_hip      = kp(11); r_hip     = kp(12)

    torso_length = _distance(shoulder, hip)
    if torso_length < 1:
        return None

    # Ángulos básicos
    back_angle_vertical = _angle_with_vertical(shoulder, hip)
    neck_angle = _angle_3p(ear, shoulder, hip)
    knee_angle = _angle_3p(hip, knee, ankle)
    hip_angle = _angle_3p(shoulder, hip, knee)

    # Distancias normalizadas
    shoulder_width = _distance(l_shoulder, r_shoulder)
    hip_width = _distance(l_hip, r_hip)
    grip_width = _distance(l_wrist, r_wrist)
    grip_to_shoulder = grip_width / shoulder_width if shoulder_width > 0 else np.nan
    grip_to_hip = grip_width / hip_width if hip_width > 0 else np.nan

    bar_point = _midpoint(l_wrist, r_wrist)
    shin_point = _midpoint(knee, ankle)
    bar_to_shin_x = abs(bar_point[0] - shin_point[0]) / torso_length

    wrist_height_ratio = (hip[1] - bar_point[1]) / torso_length
    wrist_height_ratio = float(np.clip(wrist_height_ratio,
                                       WRIST_HEIGHT_MIN, WRIST_HEIGHT_MAX))

    # Features de interacción fase-postura
    knee_x_phase = _safe_product(knee_angle, wrist_height_ratio)
    neck_x_phase = _safe_product(neck_angle, wrist_height_ratio)
    phase_low = max(0, -wrist_height_ratio)
    knee_extension_low = _safe_product(knee_angle, phase_low)

    avg_pose_conf = float(np.mean(kp_conf))

    return {
        "back_angle_vertical": back_angle_vertical,
        "neck_angle": neck_angle,
        "knee_angle": knee_angle,
        "hip_angle": hip_angle,
        "grip_to_shoulder": grip_to_shoulder,
        "grip_to_hip": grip_to_hip,
        "bar_to_shin_x": bar_to_shin_x,
        "wrist_height_ratio": wrist_height_ratio,
        "avg_pose_conf": avg_pose_conf,
        "side_detected": 1 if side_name == "right" else 0,
        "knee_x_phase": knee_x_phase,
        "neck_x_phase": neck_x_phase,
        "knee_extension_low": knee_extension_low,
    }
