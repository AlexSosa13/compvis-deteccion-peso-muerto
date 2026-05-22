"""Estimación de pose: selección de la persona principal y dibujo del esqueleto."""

import math

import cv2
import numpy as np

from .constants import (
    W_AREA, W_CENTER, W_CONF,
    MIN_KEYPOINTS, MIN_AVG_CONFIDENCE, MIN_AREA_RATIO,
    SKELETON,
)


def compute_score(bbox, kp_conf, img_w, img_h):
    """Puntuación de un candidato a persona principal."""
    x1, y1, x2, y2 = bbox
    area_ratio = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)
    bx, by = (x1 + x2) / 2, (y1 + y2) / 2
    dist = math.hypot(bx - img_w / 2, by - img_h / 2)
    diagonal = math.hypot(img_w, img_h)
    centrality = 1 - (dist / diagonal)
    valid_confs = [c for c in kp_conf if c > 0]
    avg_conf = sum(valid_confs) / len(valid_confs) if valid_confs else 0
    score = W_AREA * area_ratio + W_CENTER * centrality + W_CONF * avg_conf
    return score, area_ratio, avg_conf


def select_main_person(yolo_result, img_w, img_h):
    """Selecciona la persona principal de un resultado de YOLO-pose.

    Args:
        yolo_result: el primer elemento devuelto por model(img).
        img_w, img_h: dimensiones de la imagen.

    Returns:
        Tupla (kp_xy, kp_conf, bbox) o None si ninguna detección pasa los filtros.
    """
    if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
        return None

    bboxes = yolo_result.boxes.xyxy.cpu().numpy()
    kp_xy = yolo_result.keypoints.xy.cpu().numpy()
    kp_conf = yolo_result.keypoints.conf.cpu().numpy()

    best_score = -1
    best_idx = None
    for i in range(len(bboxes)):
        n_valid = int((kp_conf[i] > 0).sum())
        if n_valid < MIN_KEYPOINTS:
            continue
        score, area_ratio, avg_conf = compute_score(bboxes[i], kp_conf[i], img_w, img_h)
        if avg_conf < MIN_AVG_CONFIDENCE or area_ratio < MIN_AREA_RATIO:
            continue
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        return None

    return kp_xy[best_idx], kp_conf[best_idx], bboxes[best_idx]


def draw_pose_on_image(img_bgr, kp_xy, kp_conf, bbox, threshold=0.3):
    """Dibuja la bounding box y el esqueleto sobre una imagen.

    Returns:
        Imagen BGR con las anotaciones dibujadas (copia, no modifica la original).
    """
    annotated = img_bgr.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for a, b in SKELETON:
        if kp_conf[a] > threshold and kp_conf[b] > threshold:
            pa = (int(kp_xy[a][0]), int(kp_xy[a][1]))
            pb = (int(kp_xy[b][0]), int(kp_xy[b][1]))
            cv2.line(annotated, pa, pb, (255, 200, 0), 2)
    for i, (x, y) in enumerate(kp_xy):
        if kp_conf[i] > threshold:
            cv2.circle(annotated, (int(x), int(y)), 4, (0, 255, 255), -1)
    return annotated
