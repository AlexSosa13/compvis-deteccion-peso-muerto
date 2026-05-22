"""Carga de modelos y funciones de predicción para las tres ramas."""

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import joblib

from .constants import (
    LABEL_COLUMNS, TABULAR_FEATURES,
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===================== CARGA =====================

def load_xgboost(path):
    """Carga el bundle de XGBoost generado por train_xgboost.py."""
    bundle = joblib.load(path)
    return bundle["models"]


def build_cnn(num_classes):
    """Construye la arquitectura EfficientNet-B0 con la cabeza multietiqueta."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_cnn(path):
    """Carga la CNN y la deja en eval() en el dispositivo disponible."""
    checkpoint = torch.load(path, map_location=DEVICE)
    model = build_cnn(len(LABEL_COLUMNS))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE)
    model.eval()
    return model


def load_fusion_config(path):
    """Carga la configuración de fusión (pesos y umbrales por clase)."""
    config = joblib.load(path)
    return config["weights"], config["thresholds_weighted"]


# ===================== TRANSFORMACIÓN DE IMAGEN PARA LA CNN =====================

def _resize_with_padding(img_pil, target_size):
    w, h = img_pil.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
    new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    new_img.paste(img_resized, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return new_img


cnn_transform = transforms.Compose([
    transforms.Lambda(lambda img: _resize_with_padding(img, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ===================== PREDICCIÓN =====================

def predict_tabular(xgb_models, features_dict):
    """Devuelve dict clase -> probabilidad usando los modelos XGBoost."""
    X = np.array([[features_dict[f] for f in TABULAR_FEATURES]], dtype=np.float32)
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0)
    return {c: float(xgb_models[c].predict_proba(X)[0, 1]) for c in LABEL_COLUMNS}


def predict_cnn(cnn_model, img_bgr):
    """Devuelve dict clase -> probabilidad pasando la imagen BGR por la CNN."""
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    tensor = cnn_transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = cnn_model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return {LABEL_COLUMNS[i]: float(probs[i]) for i in range(len(LABEL_COLUMNS))}


def fuse_probabilities(p_tab, p_cnn, weights):
    """Combina probabilidades por promedio ponderado por clase.

    weights: dict clase -> w_tabular. El peso de la CNN es (1 - w).
    """
    return {c: weights[c] * p_tab[c] + (1 - weights[c]) * p_cnn[c]
            for c in LABEL_COLUMNS}


def apply_thresholds(probs_fused, thresholds):
    """Devuelve dict clase -> bool (True si supera su umbral)."""
    return {c: probs_fused[c] >= thresholds[c] for c in LABEL_COLUMNS}
