"""Constantes compartidas del proyecto."""

# Clases de error
LABEL_COLUMNS = ["Agarre", "Cabeza", "Distancia", "Dorsal", "Lumbar", "Pierna"]

# Features de la rama tabular (en el mismo orden que durante el entrenamiento)
TABULAR_FEATURES = [
    "back_angle_vertical", "neck_angle", "knee_angle", "hip_angle",
    "grip_to_shoulder", "grip_to_hip", "bar_to_shin_x",
    "wrist_height_ratio", "avg_pose_conf", "side_detected",
    "knee_x_phase", "neck_x_phase", "knee_extension_low",
]

# Preprocesado: eliminación de bandas negras
BORDER_BRIGHTNESS_THRESHOLD = 15
MIN_USEFUL_FRACTION = 0.3

# Heurística de selección de la persona principal
W_AREA = 0.5
W_CENTER = 0.3
W_CONF = 0.2
MIN_KEYPOINTS = 8
MIN_AVG_CONFIDENCE = 0.45
MIN_AREA_RATIO = 0.03

# Clip para wrist_height_ratio (evita contaminar features de interacción)
WRIST_HEIGHT_MIN = -1.5
WRIST_HEIGHT_MAX = 0.5

# Keypoints COCO de YOLO-pose
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Conexiones del esqueleto para dibujarlo
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]

# CNN
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
