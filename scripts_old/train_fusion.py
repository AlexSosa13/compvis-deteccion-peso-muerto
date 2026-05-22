"""
Modelo de fusión: combina las predicciones de la rama tabular (XGBoost sobre
features de pose) y la rama de imagen (CNN EfficientNet-B0).

Las dos ramas capturan información complementaria:
  - La rama tabular es fuerte en errores geométricos claros (p. ej. Cabeza).
  - La rama CNN es fuerte en errores con señal visual fina (Agarre, Distancia).

Se prueban tres estrategias de fusión:
  1. Promedio simple de probabilidades.
  2. Promedio ponderado por clase (peso calibrado en validation).
  3. Stacking: regresión logística por clase sobre las 12 probabilidades base.

IMPORTANTE - evitar fuga de datos:
  El meta-modelo se calibra/entrena sobre VALIDATION, no sobre train, porque
  las ramas base ya se entrenaron con train y sus predicciones sobre train
  estarían infladas. La evaluación final es sobre TEST.

Requisitos previos:
  - xgboost_models.joblib  (de train_xgboost.py)
  - efficientnet_b0_deadlift.pt  (de train_cnn.py)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, precision_recall_curve, average_precision_score
)

# ---------- CONFIGURACIÓN ----------
IMAGES_DIR = "data"
VAL_CSV = "splits/dataset_val.csv"
TEST_CSV = "splits/dataset_test.csv"

XGBOOST_MODEL = "output/resultados_xgboost/xgboost_models.joblib"
CNN_MODEL = "output/resultados_cnn/efficientnet_b0_deadlift.pt"

OUTPUT_DIR = "output/resultados_fusion"

LABEL_COLUMNS = ["Agarre", "Cabeza", "Distancia", "Dorsal", "Lumbar", "Pierna"]
TABULAR_FEATURES = [
    "back_angle_vertical", "neck_angle",
    "knee_angle", "hip_angle", "grip_to_shoulder", "grip_to_hip",
    "bar_to_shin_x", "wrist_height_ratio", "avg_pose_conf", "side_detected",
    "knee_x_phase", "neck_x_phase", "knee_extension_low",
]

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Resolución del barrido de pesos para la fusión ponderada
WEIGHT_GRID = np.arange(0.0, 1.01, 0.05)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# -----------------------------------


# ===================== PREDICCIONES DE LA RAMA TABULAR =====================

def get_tabular_probas(csv_path):
    """Carga el modelo XGBoost y devuelve probabilidades por clase."""
    bundle = joblib.load(XGBOOST_MODEL)
    models_dict = bundle["models"]
    feature_columns = bundle["feature_columns"]

    df = pd.read_csv(csv_path)
    X = df[feature_columns].values.astype(np.float32)
    # Imputar NaN residuales
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        if np.isnan(col).any():
            X[np.isnan(col), col_idx] = np.nanmedian(col)

    probas = np.zeros((len(df), len(LABEL_COLUMNS)), dtype=np.float32)
    for i, name in enumerate(LABEL_COLUMNS):
        probas[:, i] = models_dict[name].predict_proba(X)[:, 1]

    labels = df[LABEL_COLUMNS].values.astype(np.float32)
    filenames = df["filename"].tolist()
    return probas, labels, filenames


# ===================== PREDICCIONES DE LA RAMA CNN =====================

def resize_with_padding(img_size):
    def _transform(img):
        w, h = img.size
        scale = img_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        new_img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
        new_img.paste(img, ((img_size - new_w) // 2, (img_size - new_h) // 2))
        return new_img
    return _transform


eval_transform = transforms.Compose([
    transforms.Lambda(resize_with_padding(IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class ImageDataset(Dataset):
    """Dataset que carga solo imágenes en el orden de un CSV dado."""

    def __init__(self, csv_path, images_dir, transform):
        self.df = pd.read_csv(csv_path)
        if "is_synthetic" in self.df.columns:
            self.df = self.df[self.df["is_synthetic"] == 0].reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.images_dir / row["filename"]).convert("RGB")
        img = self.transform(img)
        return img, idx  # idx para mantener el orden


def build_cnn(num_classes):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_cnn_probas(csv_path):
    """Carga la CNN y devuelve probabilidades por clase en el orden del CSV."""
    checkpoint = torch.load(CNN_MODEL, map_location=DEVICE)
    model = build_cnn(len(LABEL_COLUMNS))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE)
    model.eval()

    dataset = ImageDataset(csv_path, IMAGES_DIR, eval_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    probas = np.zeros((len(dataset), len(LABEL_COLUMNS)), dtype=np.float32)
    with torch.no_grad():
        for images, idxs in loader:
            images = images.to(DEVICE)
            logits = model(images)
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            for j, idx in enumerate(idxs.numpy()):
                probas[idx] = batch_probs[j]

    return probas


# ===================== ESTRATEGIAS DE FUSIÓN =====================

def best_f1_and_threshold(y_true, y_proba):
    """Mejor F1 variando el umbral, y el umbral correspondiente."""
    precs, recs, thresholds = precision_recall_curve(y_true, y_proba)
    denom = precs + recs
    with np.errstate(divide="ignore", invalid="ignore"):
        f1s = np.where(denom > 0, 2 * precs * recs / denom, 0)
    if len(thresholds) == 0:
        return 0.0, 0.5
    best_idx = np.argmax(f1s[:-1])
    return float(f1s[best_idx]), float(thresholds[best_idx])


def fusion_simple_average(proba_tab, proba_cnn):
    """Estrategia 1: promedio simple."""
    return (proba_tab + proba_cnn) / 2


def fit_weighted_average(proba_tab_val, proba_cnn_val, y_val, label_names):
    """Estrategia 2: busca el peso óptimo por clase sobre validation.

    Para cada clase, prueba w en WEIGHT_GRID y se queda con el que maximiza F1.
    fusion = w * tabular + (1-w) * cnn
    """
    weights = {}
    thresholds = {}
    for i, name in enumerate(label_names):
        best_f1 = -1
        best_w = 0.5
        best_thr = 0.5
        for w in WEIGHT_GRID:
            fused = w * proba_tab_val[:, i] + (1 - w) * proba_cnn_val[:, i]
            f1, thr = best_f1_and_threshold(y_val[:, i], fused)
            if f1 > best_f1:
                best_f1 = f1
                best_w = w
                best_thr = thr
        weights[name] = best_w
        thresholds[name] = best_thr
    return weights, thresholds


def apply_weighted_average(proba_tab, proba_cnn, weights, label_names):
    """Aplica los pesos por clase."""
    fused = np.zeros_like(proba_tab)
    for i, name in enumerate(label_names):
        w = weights[name]
        fused[:, i] = w * proba_tab[:, i] + (1 - w) * proba_cnn[:, i]
    return fused


def fit_stacking(proba_tab_val, proba_cnn_val, y_val, label_names):
    """Estrategia 3: stacking con regresión logística por clase.

    Cada meta-clasificador recibe las 12 probabilidades base (6 tab + 6 cnn).
    Regularización fuerte (C bajo) porque validation es pequeño.
    """
    X_meta = np.hstack([proba_tab_val, proba_cnn_val])  # (n, 12)
    meta_models = {}
    thresholds = {}
    for i, name in enumerate(label_names):
        clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000)
        clf.fit(X_meta, y_val[:, i])
        meta_models[name] = clf
        # Umbral óptimo sobre val
        proba = clf.predict_proba(X_meta)[:, 1]
        _, thr = best_f1_and_threshold(y_val[:, i], proba)
        thresholds[name] = thr
    return meta_models, thresholds


def apply_stacking(proba_tab, proba_cnn, meta_models, label_names):
    X_meta = np.hstack([proba_tab, proba_cnn])
    fused = np.zeros((X_meta.shape[0], len(label_names)), dtype=np.float32)
    for i, name in enumerate(label_names):
        fused[:, i] = meta_models[name].predict_proba(X_meta)[:, 1]
    return fused


# ===================== EVALUACIÓN =====================

def evaluate(proba, y_true, thresholds, label_names):
    """Métricas por clase con umbrales (dict por clase)."""
    thr_arr = np.array([thresholds[c] for c in label_names])
    preds = np.zeros_like(y_true, dtype=int)
    for i in range(len(label_names)):
        preds[:, i] = (proba[:, i] >= thr_arr[i]).astype(int)

    rows = []
    for i, name in enumerate(label_names):
        rows.append({
            "label": name,
            "threshold": thr_arr[i],
            "support": int(y_true[:, i].sum()),
            "precision": precision_score(y_true[:, i], preds[:, i], zero_division=0),
            "recall": recall_score(y_true[:, i], preds[:, i], zero_division=0),
            "f1": f1_score(y_true[:, i], preds[:, i], zero_division=0),
            "ap": average_precision_score(y_true[:, i], proba[:, i]),
        })
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, preds, average="micro", zero_division=0)
    return pd.DataFrame(rows), macro_f1, micro_f1, preds


def plot_confusion_matrices(y_true, preds, label_names, output_path):
    n = len(label_names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows))
    axes = axes.flatten()
    for i, name in enumerate(label_names):
        cm = confusion_matrix(y_true[:, i], preds[:, i])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[f"{name}=0", f"{name}=1"],
                    yticklabels=[f"{name}=0", f"{name}=1"],
                    cbar=False, ax=axes[i])
        axes[i].set_xlabel("Predicho")
        axes[i].set_ylabel("Real")
        axes[i].set_title(name)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Dispositivo: {DEVICE}")

    # ===== Predicciones base de cada rama =====
    print("\nObteniendo predicciones de la rama tabular (XGBoost)...")
    proba_tab_val, y_val, files_val = get_tabular_probas(VAL_CSV)
    proba_tab_test, y_test, files_test = get_tabular_probas(TEST_CSV)

    print("Obteniendo predicciones de la rama CNN (EfficientNet-B0)...")
    proba_cnn_val = get_cnn_probas(VAL_CSV)
    proba_cnn_test = get_cnn_probas(TEST_CSV)

    # Verificación de alineación: ambas ramas deben usar el mismo orden de filas
    assert proba_tab_val.shape == proba_cnn_val.shape, \
        "Desalineación entre ramas en VAL. ¿Mismo CSV / mismo filtrado?"
    assert proba_tab_test.shape == proba_cnn_test.shape, \
        "Desalineación entre ramas en TEST."

    results = {}

    # ===== Estrategia 1: promedio simple =====
    print("\n" + "=" * 60)
    print("ESTRATEGIA 1: Promedio simple")
    print("=" * 60)
    fused_val = fusion_simple_average(proba_tab_val, proba_cnn_val)
    fused_test = fusion_simple_average(proba_tab_test, proba_cnn_test)
    # Umbrales óptimos sobre val
    thr_simple = {}
    for i, name in enumerate(LABEL_COLUMNS):
        _, thr = best_f1_and_threshold(y_val[:, i], fused_val[:, i])
        thr_simple[name] = thr
    m1, macro1, micro1, _ = evaluate(fused_test, y_test, thr_simple, LABEL_COLUMNS)
    print(m1.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n  Macro-F1: {macro1:.3f}, Micro-F1: {micro1:.3f}")
    results["promedio_simple"] = macro1

    # ===== Estrategia 2: promedio ponderado por clase =====
    print("\n" + "=" * 60)
    print("ESTRATEGIA 2: Promedio ponderado por clase")
    print("=" * 60)
    weights, thr_weighted = fit_weighted_average(
        proba_tab_val, proba_cnn_val, y_val, LABEL_COLUMNS)
    print("Pesos óptimos (w = peso de la rama TABULAR, 1-w = peso CNN):")
    for name in LABEL_COLUMNS:
        print(f"  {name}: w_tabular={weights[name]:.2f}, w_cnn={1-weights[name]:.2f}")
    fused_test_w = apply_weighted_average(
        proba_tab_test, proba_cnn_test, weights, LABEL_COLUMNS)
    m2, macro2, micro2, preds2 = evaluate(fused_test_w, y_test, thr_weighted, LABEL_COLUMNS)
    print("\n", m2.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n  Macro-F1: {macro2:.3f}, Micro-F1: {micro2:.3f}")
    results["promedio_ponderado"] = macro2

    # ===== Estrategia 3: stacking =====
    print("\n" + "=" * 60)
    print("ESTRATEGIA 3: Stacking (regresión logística por clase)")
    print("=" * 60)
    meta_models, thr_stack = fit_stacking(
        proba_tab_val, proba_cnn_val, y_val, LABEL_COLUMNS)
    fused_test_s = apply_stacking(
        proba_tab_test, proba_cnn_test, meta_models, LABEL_COLUMNS)
    m3, macro3, micro3, preds3 = evaluate(fused_test_s, y_test, thr_stack, LABEL_COLUMNS)
    print(m3.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n  Macro-F1: {macro3:.3f}, Micro-F1: {micro3:.3f}")
    results["stacking"] = macro3

    # ===== Comparativa final =====
    print("\n" + "#" * 60)
    print("# COMPARATIVA FINAL")
    print("#" * 60)
    print(f"\n  Promedio simple:    Macro-F1 = {results['promedio_simple']:.3f}")
    print(f"  Promedio ponderado: Macro-F1 = {results['promedio_ponderado']:.3f}")
    print(f"  Stacking:           Macro-F1 = {results['stacking']:.3f}")

    best_strategy = max(results, key=results.get)
    print(f"\n  Mejor estrategia: {best_strategy} ({results[best_strategy]:.3f})")

    # Guardar resultados de la mejor estrategia
    if best_strategy == "promedio_ponderado":
        m2.to_csv(out / "metrics_test_fusion.csv", index=False)
        plot_confusion_matrices(y_test, preds2, LABEL_COLUMNS,
                                out / "confusion_matrices_fusion.png")
    elif best_strategy == "stacking":
        m3.to_csv(out / "metrics_test_fusion.csv", index=False)
        plot_confusion_matrices(y_test, preds3, LABEL_COLUMNS,
                                out / "confusion_matrices_fusion.png")
    else:
        m1.to_csv(out / "metrics_test_fusion.csv", index=False)

    # Guardar configuración de fusión
    joblib.dump({
        "best_strategy": best_strategy,
        "weights": weights,
        "thresholds_weighted": thr_weighted,
        "thresholds_simple": thr_simple,
        "thresholds_stacking": thr_stack,
        "meta_models": meta_models,
        "label_columns": LABEL_COLUMNS,
    }, out / "fusion_config.joblib")

    print(f"\nResultados guardados en {out.absolute()}")


if __name__ == "__main__":
    main()