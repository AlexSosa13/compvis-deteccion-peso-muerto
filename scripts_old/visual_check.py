"""
Script de verificación visual del pipeline de extracción de pose.

Toma una muestra aleatoria de imágenes y genera versiones anotadas con:
  - Bounding box de la persona seleccionada (verde) con su score
  - Bounding boxes descartadas (rojo) con el motivo del descarte
  - Esqueleto y keypoints de la persona principal
  - Punto central de la imagen (referencia para la centralidad)

Aplica el mismo auto-crop de bandas negras que extract_keypoints.py para que
la verificación refleje fielmente lo que verá el script principal.
"""

import math
import random
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# ---------- CONFIGURACIÓN ----------
IMAGES_DIR = "data"
OUTPUT_DIR = "output/verificacion_visual"
MODEL_NAME = "yolo11n-pose.pt"

NUM_SAMPLES = 80
RANDOM_SEED = 42

# Auto-crop (debe coincidir con extract_keypoints.py)
AUTO_CROP_BORDERS = False
BORDER_BRIGHTNESS_THRESHOLD = 15
MIN_USEFUL_FRACTION = 0.3

# Heurística (debe coincidir con extract_keypoints.py)
W_AREA = 0.5
W_CENTER = 0.3
W_CONF = 0.2

MIN_KEYPOINTS = 8
MIN_AVG_CONFIDENCE = 0.3
MIN_AREA_RATIO = 0.05

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]

COLOR_SELECTED = (0, 255, 0)
COLOR_DISCARDED = (0, 0, 255)
COLOR_NOT_CHOSEN = (0, 165, 255)
COLOR_SKELETON = (255, 200, 0)
COLOR_KEYPOINT = (0, 255, 255)
COLOR_CENTER = (255, 0, 255)
# -----------------------------------


def crop_black_borders(img, threshold=BORDER_BRIGHTNESS_THRESHOLD):
    if img is None or img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    col_means = gray.mean(axis=0)
    row_means = gray.mean(axis=1)
    non_black_cols = np.where(col_means > threshold)[0]
    non_black_rows = np.where(row_means > threshold)[0]
    if len(non_black_cols) == 0 or len(non_black_rows) == 0:
        return img
    x1 = int(non_black_cols[0])
    x2 = int(non_black_cols[-1]) + 1
    y1 = int(non_black_rows[0])
    y2 = int(non_black_rows[-1]) + 1
    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        return img
    useful_fraction = (cropped.shape[0] * cropped.shape[1]) / (h * w)
    if useful_fraction < MIN_USEFUL_FRACTION:
        return img
    return cropped


def compute_score(bbox, keypoints_conf, img_w, img_h):
    x1, y1, x2, y2 = bbox
    bbox_area = (x2 - x1) * (y2 - y1)
    img_area = img_w * img_h
    area_norm = bbox_area / img_area
    bbox_cx = (x1 + x2) / 2
    bbox_cy = (y1 + y2) / 2
    img_cx = img_w / 2
    img_cy = img_h / 2
    dist = math.sqrt((bbox_cx - img_cx) ** 2 + (bbox_cy - img_cy) ** 2)
    diagonal = math.sqrt(img_w ** 2 + img_h ** 2)
    centrality = 1 - (dist / diagonal)
    valid_confs = [c for c in keypoints_conf if c > 0]
    avg_conf = sum(valid_confs) / len(valid_confs) if valid_confs else 0
    score = W_AREA * area_norm + W_CENTER * centrality + W_CONF * avg_conf
    return score, area_norm, avg_conf, centrality


def evaluate_detection(bbox, kp_conf, img_w, img_h):
    num_valid_kp = int((kp_conf > 0).sum())
    score, area_ratio, avg_conf, centrality = compute_score(bbox, kp_conf, img_w, img_h)
    info = {
        "score": score, "area_ratio": area_ratio,
        "avg_conf": avg_conf, "centrality": centrality,
        "num_valid_kp": num_valid_kp,
    }
    reason = None
    if num_valid_kp < MIN_KEYPOINTS:
        reason = f"KP<{MIN_KEYPOINTS}"
    elif avg_conf < MIN_AVG_CONFIDENCE:
        reason = f"conf<{MIN_AVG_CONFIDENCE}"
    elif area_ratio < MIN_AREA_RATIO:
        reason = f"area<{MIN_AREA_RATIO}"
    return info, reason


def draw_skeleton(img, kp_xy, kp_conf, threshold=0.3):
    for a, b in SKELETON:
        if kp_conf[a] > threshold and kp_conf[b] > threshold:
            pt_a = (int(kp_xy[a][0]), int(kp_xy[a][1]))
            pt_b = (int(kp_xy[b][0]), int(kp_xy[b][1]))
            cv2.line(img, pt_a, pt_b, COLOR_SKELETON, 2)
    for i, (x, y) in enumerate(kp_xy):
        if kp_conf[i] > threshold:
            cv2.circle(img, (int(x), int(y)), 4, COLOR_KEYPOINT, -1)


def draw_bbox_with_label(img, bbox, color, label, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def annotate_image(img, result):
    img_h, img_w = img.shape[:2]
    annotated = img.copy()
    cv2.drawMarker(annotated, (img_w // 2, img_h // 2), COLOR_CENTER,
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    if result.boxes is None or len(result.boxes) == 0:
        cv2.putText(annotated, "Sin detecciones", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return annotated

    bboxes = result.boxes.xyxy.cpu().numpy()
    kp_xy = result.keypoints.xy.cpu().numpy()
    kp_conf = result.keypoints.conf.cpu().numpy()

    evaluations = []
    for i in range(len(bboxes)):
        info, reason = evaluate_detection(bboxes[i], kp_conf[i], img_w, img_h)
        evaluations.append((i, info, reason))

    valid_evals = [(i, info) for i, info, reason in evaluations if reason is None]
    selected_idx = None
    if valid_evals:
        selected_idx = max(valid_evals, key=lambda x: x[1]["score"])[0]

    for i, info, reason in evaluations:
        if i == selected_idx:
            continue
        if reason is None:
            label = f"#{i} score={info['score']:.2f} (no elegida)"
            color = COLOR_NOT_CHOSEN
        else:
            label = f"#{i} descartada: {reason}"
            color = COLOR_DISCARDED
        draw_bbox_with_label(annotated, bboxes[i], color, label, thickness=2)

    if selected_idx is not None:
        info = next(inf for i, inf, _ in evaluations if i == selected_idx)
        label = (f"PRINCIPAL  score={info['score']:.2f} "
                 f"a={info['area_ratio']:.2f} c={info['centrality']:.2f} "
                 f"conf={info['avg_conf']:.2f}")
        draw_bbox_with_label(annotated, bboxes[selected_idx], COLOR_SELECTED,
                             label, thickness=3)
        draw_skeleton(annotated, kp_xy[selected_idx], kp_conf[selected_idx])
    else:
        cv2.putText(annotated, "Ninguna deteccion valida", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return annotated


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando modelo {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    all_images = sorted([
        p for p in Path(IMAGES_DIR).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    print(f"Encontradas {len(all_images)} imágenes")

    random.seed(RANDOM_SEED)
    sample = random.sample(all_images, min(NUM_SAMPLES, len(all_images)))
    print(f"Procesando muestra de {len(sample)} imágenes...")

    for i, img_path in enumerate(sample):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if AUTO_CROP_BORDERS:
            img = crop_black_borders(img)

        results = model(img, verbose=False)
        annotated = annotate_image(img, results[0])

        out_path = out_dir / f"check_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), annotated)

        if (i + 1) % 20 == 0:
            print(f"  Procesadas {i + 1}/{len(sample)}")

    print(f"\nListo. Revisa las imágenes en: {out_dir.absolute()}")
    print("\nLeyenda:")
    print("  - Cuadro VERDE grueso: persona principal seleccionada")
    print("  - Cuadro NARANJA: detección válida pero no elegida (otra persona)")
    print("  - Cuadro ROJO: detección descartada (con motivo)")
    print("  - Cruz MAGENTA: centro de la imagen (referencia de centralidad)")
    print("  - Esqueleto CIAN + puntos AMARILLOS: keypoints de la persona principal")


if __name__ == "__main__":
    main()
