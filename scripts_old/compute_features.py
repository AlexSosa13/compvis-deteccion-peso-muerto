"""
Calcula features interpretables a partir de los keypoints extraídos.

Cambios respecto a la versión anterior:
  - Clip de outliers en wrist_height_ratio para evitar contaminar las features
    derivadas (poses degeneradas con torso_length casi nulo daban valores absurdos).
  - Reducido el número de features de interacción a las 3 más informativas según
    el análisis exploratorio: knee_x_phase, neck_x_phase, knee_extension_low.
    Esto evita la multicolinealidad entre productos similares y deja al modelo
    más foco para clases con pocos ejemplos.
"""

import math
import pandas as pd
import numpy as np

# ---------- CONFIGURACIÓN ----------
KEYPOINTS_CSV = "output/keypoints.csv"
LABELS_CSV = "data/_classes.csv"
OUTPUT_CSV = "output/dataset_features.csv"

DROP_BIEN = True
LABEL_COLUMNS = ["Agarre", "Cabeza", "Distancia", "Dorsal", "Lumbar", "Pierna"]

# Límites para clip de outliers (basados en valores físicamente plausibles)
WRIST_HEIGHT_MIN = -1.5
WRIST_HEIGHT_MAX = 0.5
# -----------------------------------


def get_kp(row, name):
    return row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_conf"]


def angle_3p(p1, p2, p3):
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return np.nan
    cos = max(-1, min(1, dot / (n1 * n2)))
    return math.degrees(math.acos(cos))


def angle_with_vertical(p_top, p_bottom):
    dx = p_top[0] - p_bottom[0]
    dy = p_top[1] - p_bottom[1]
    if dx == 0 and dy == 0:
        return np.nan
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def detect_side(row):
    left_kps = ["left_shoulder", "left_hip", "left_knee", "left_ankle", "left_wrist"]
    right_kps = ["right_shoulder", "right_hip", "right_knee", "right_ankle", "right_wrist"]
    left_conf = np.mean([row[f"{k}_conf"] for k in left_kps])
    right_conf = np.mean([row[f"{k}_conf"] for k in right_kps])
    return "left" if left_conf >= right_conf else "right"


def safe_product(*values):
    """Producto que devuelve NaN si cualquier componente lo es."""
    if any(np.isnan(v) for v in values):
        return np.nan
    result = 1.0
    for v in values:
        result *= v
    return result


def compute_features(row):
    if row["pose_valid"] == 0:
        return None

    side = detect_side(row)

    shoulder = get_kp(row, f"{side}_shoulder")[:2]
    hip = get_kp(row, f"{side}_hip")[:2]
    knee = get_kp(row, f"{side}_knee")[:2]
    ankle = get_kp(row, f"{side}_ankle")[:2]
    ear = get_kp(row, f"{side}_ear")[:2]

    l_shoulder = get_kp(row, "left_shoulder")[:2]
    r_shoulder = get_kp(row, "right_shoulder")[:2]
    l_wrist = get_kp(row, "left_wrist")[:2]
    r_wrist = get_kp(row, "right_wrist")[:2]
    l_hip = get_kp(row, "left_hip")[:2]
    r_hip = get_kp(row, "right_hip")[:2]

    torso_length = distance(shoulder, hip)
    if torso_length < 1:
        return None

    # --- Ángulos básicos ---
    back_angle_vertical = angle_with_vertical(shoulder, hip)
    neck_angle = angle_3p(ear, shoulder, hip)
    knee_angle = angle_3p(hip, knee, ankle)
    hip_angle = angle_3p(shoulder, hip, knee)

    # --- Distancias normalizadas ---
    shoulder_width = distance(l_shoulder, r_shoulder)
    hip_width = distance(l_hip, r_hip)
    grip_width = distance(l_wrist, r_wrist)

    grip_to_shoulder = grip_width / shoulder_width if shoulder_width > 0 else np.nan
    grip_to_hip = grip_width / hip_width if hip_width > 0 else np.nan

    bar_point = midpoint(l_wrist, r_wrist)
    shin_point = midpoint(knee, ankle)
    bar_to_shin_x = abs(bar_point[0] - shin_point[0]) / torso_length

    # --- Proxy de fase del movimiento con clip de outliers ---
    wrist_height_ratio = (hip[1] - bar_point[1]) / torso_length
    # Clip para evitar que poses degeneradas (con valores absurdos) contaminen
    # las features de interacción
    wrist_height_ratio = float(np.clip(wrist_height_ratio, WRIST_HEIGHT_MIN, WRIST_HEIGHT_MAX))

    # --- Features de interacción fase-postura (reducidas a las más informativas) ---
    # Producto knee x phase: captura "pierna recta abajo" o "flexionada arriba"
    knee_x_phase = safe_product(knee_angle, wrist_height_ratio)
    # Producto neck x phase: cuello mal posicionado depende de la fase
    neck_x_phase = safe_product(neck_angle, wrist_height_ratio)

    # Anomalía específica: pierna extendida en fase baja
    phase_low = max(0, -wrist_height_ratio)
    knee_extension_low = safe_product(knee_angle, phase_low)

    # Confianza global de la pose
    conf_cols = [c for c in row.index if c.endswith("_conf")]
    avg_pose_conf = np.mean([row[c] for c in conf_cols])

    return {
        # Básicas (10)
        "back_angle_vertical": back_angle_vertical,
        "neck_angle": neck_angle,
        "knee_angle": knee_angle,
        "hip_angle": hip_angle,
        "grip_to_shoulder": grip_to_shoulder,
        "grip_to_hip": grip_to_hip,
        "bar_to_shin_x": bar_to_shin_x,
        "wrist_height_ratio": wrist_height_ratio,
        "avg_pose_conf": avg_pose_conf,
        "side_detected": 1 if side == "right" else 0,
        # Interacción fase-postura (3 reducidas)
        "knee_x_phase": knee_x_phase,
        "neck_x_phase": neck_x_phase,
        "knee_extension_low": knee_extension_low,
    }


def main():
    kp_df = pd.read_csv(KEYPOINTS_CSV)
    labels_df = pd.read_csv(LABELS_CSV)
    labels_df.columns = labels_df.columns.str.strip()

    if DROP_BIEN and "Bien" in labels_df.columns:
        labels_df = labels_df.drop(columns=["Bien"])

    print(f"Keypoints: {len(kp_df)} filas")
    print(f"Labels: {len(labels_df)} filas")

    feature_names = [
        "back_angle_vertical", "neck_angle", "knee_angle", "hip_angle",
        "grip_to_shoulder", "grip_to_hip", "bar_to_shin_x", "wrist_height_ratio",
        "avg_pose_conf", "side_detected",
        "knee_x_phase", "neck_x_phase", "knee_extension_low",
    ]

    features_list = []
    for _, row in kp_df.iterrows():
        feats = compute_features(row)
        if feats is None:
            feats = {k: np.nan for k in feature_names}
        feats["filename"] = row["filename"]
        feats["pose_valid"] = row["pose_valid"]
        features_list.append(feats)

    feats_df = pd.DataFrame(features_list)

    merged = feats_df.merge(labels_df, on="filename", how="inner")
    print(f"\nDataset final: {len(merged)} filas")
    print(f"Pose válida en {int(merged['pose_valid'].sum())} frames "
          f"({100 * merged['pose_valid'].mean():.1f}%)")

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"Guardado en {OUTPUT_CSV}")

    print(f"\nNúmero total de features: {len(feature_names)}")
    print("\nDistribución de etiquetas:")
    for col in LABEL_COLUMNS:
        if col in merged.columns:
            print(f"  {col}: {int(merged[col].sum())}")

    # Verificación: estadísticas de las features tras clip
    print("\nEstadísticas de features de fase (deberían estar en rangos razonables):")
    for f in ["wrist_height_ratio", "knee_x_phase", "neck_x_phase", "knee_extension_low"]:
        if f in merged.columns:
            vals = merged[f].dropna()
            print(f"  {f}: min={vals.min():.2f}, max={vals.max():.2f}, "
                  f"mean={vals.mean():.2f}, std={vals.std():.2f}")


if __name__ == "__main__":
    main()