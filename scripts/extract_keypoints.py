"""
Extrae los keypoints de la persona principal en cada imagen usando YOLO11-pose.

Incluye auto-crop de bandas negras (letterbox/pillarbox) antes de procesar:
detecta filas/columnas casi negras en los bordes y las elimina. Esto evita
que el filtro de área descarte erróneamente personas que ocupan una porción
pequeña de la imagen total pero grande del contenido útil.

La persona principal se identifica con una heurística que combina:
  - Tamaño del bounding box
  - Centralidad
  - Confianza media de la detección
"""

import csv
import math
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# ---------- CONFIGURACIÓN ----------
IMAGES_DIR = "train_clean"   # AJUSTAR
OUTPUT_CSV = "output/keypoints.csv"
MODEL_NAME = "yolo11n-pose.pt"

# Auto-crop de bandas negras
AUTO_CROP_BORDERS = False
BORDER_BRIGHTNESS_THRESHOLD = 15  # filas/columnas con media < este valor se consideran banda
MIN_USEFUL_FRACTION = 0.3         # si tras recortar queda menos del 30%, asumimos fallo y no recortamos

# Heurística persona principal
W_AREA = 0.5
W_CENTER = 0.3
W_CONF = 0.2

# Filtros
MIN_KEYPOINTS = 8
MIN_AVG_CONFIDENCE = 0.3
MIN_AREA_RATIO = 0.05

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
# -----------------------------------


def crop_black_borders(img, threshold=BORDER_BRIGHTNESS_THRESHOLD):
    """Elimina bandas negras de los bordes. Devuelve la imagen recortada."""
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
    return score, area_norm, avg_conf


def select_main_person(result, img_w, img_h):
    if result.boxes is None or len(result.boxes) == 0:
        return None

    bboxes = result.boxes.xyxy.cpu().numpy()
    kp_xy = result.keypoints.xy.cpu().numpy()
    kp_conf = result.keypoints.conf.cpu().numpy()

    best_score = -1
    best_idx = None

    for i in range(len(bboxes)):
        num_valid_kp = int((kp_conf[i] > 0).sum())
        if num_valid_kp < MIN_KEYPOINTS:
            continue

        score, area_ratio, avg_conf = compute_score(bboxes[i], kp_conf[i], img_w, img_h)

        if avg_conf < MIN_AVG_CONFIDENCE:
            continue
        if area_ratio < MIN_AREA_RATIO:
            continue

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        return None

    return kp_xy[best_idx], kp_conf[best_idx], bboxes[best_idx]


def main():
    print(f"Cargando modelo {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    image_paths = sorted([
        p for p in Path(IMAGES_DIR).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    print(f"Encontradas {len(image_paths)} imágenes")
    print(f"Auto-crop de bandas negras: {'ACTIVADO' if AUTO_CROP_BORDERS else 'DESACTIVADO'}")

    header = ["filename"]
    for kp in KEYPOINT_NAMES:
        header += [f"{kp}_x", f"{kp}_y", f"{kp}_conf"]
    header += ["img_w", "img_h", "pose_valid"]

    cropped_count = 0

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i, img_path in enumerate(image_paths):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  Aviso: no se pudo leer {img_path.name}")
                continue

            if AUTO_CROP_BORDERS:
                orig_shape = img.shape
                img = crop_black_borders(img)
                if img.shape != orig_shape:
                    cropped_count += 1

            results = model(img, verbose=False)
            result = results[0]
            img_h, img_w = img.shape[:2]

            selection = select_main_person(result, img_w, img_h)

            row = [img_path.name]
            if selection is None:
                for _ in KEYPOINT_NAMES:
                    row += [0, 0, 0]
                row += [img_w, img_h, 0]
            else:
                kp_xy, kp_conf, _ = selection
                for j in range(len(KEYPOINT_NAMES)):
                    row += [float(kp_xy[j][0]), float(kp_xy[j][1]), float(kp_conf[j])]
                row += [img_w, img_h, 1]

            writer.writerow(row)

            if (i + 1) % 100 == 0:
                print(f"  Procesadas {i + 1}/{len(image_paths)}")

    print(f"\nImágenes recortadas: {cropped_count}/{len(image_paths)}")
    print(f"Keypoints guardados en {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
