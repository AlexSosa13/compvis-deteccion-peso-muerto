"""
Baseline para clasificación multietiqueta de errores en peso muerto.

Modelo: Random Forest multi-output, un clasificador binario por clase.
Estrategia de desbalance: class_weight='balanced' (cada clase calcula sus
pesos inversamente proporcionales a su frecuencia).

Evalúa sobre validation y test, reportando por clase:
  - Precision, recall, F1
  - Matriz de confusión
  - Curva precision-recall y umbral óptimo
  - Macro-F1, micro-F1
  - Comparación de umbrales por clase

Salida: métricas en consola + CSVs + gráficos en una carpeta de resultados.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend no interactivo, evita conflicto con threads de sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
import joblib

# ---------- CONFIGURACIÓN ----------
TRAIN_CSV = "splits/dataset_train.csv"
VAL_CSV = "splits/dataset_val.csv"
TEST_CSV = "splits/dataset_test.csv"
OUTPUT_DIR = "output/results_baseline"

LABEL_COLUMNS = ["Agarre", "Cabeza", "Distancia", "Dorsal", "Lumbar", "Pierna"]

# Features a usar (eliminamos torso_horizontal porque tiene correlación -1 con back_angle_vertical)
FEATURE_COLUMNS = [
    "back_angle_vertical", "neck_angle",
    "knee_angle", "hip_angle", "grip_to_shoulder", "grip_to_hip",
    "bar_to_shin_x", "wrist_height_ratio", "avg_pose_conf", "side_detected",
    "knee_x_phase", "neck_x_phase", "knee_extension_low",
]

# Hiperparámetros del Random Forest
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,           # sin limitar
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "class_weight": "balanced",  # clave para clases desbalanceadas
    "n_jobs": -1,
    "random_state": 42,
}

SAVE_MODEL = True
# -----------------------------------


def load_split(path):
    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS].values
    y = df[LABEL_COLUMNS].values
    # Las features pueden tener algún NaN residual: imputar con la mediana del propio dataframe
    for col_idx, col in enumerate(FEATURE_COLUMNS):
        col_data = X[:, col_idx]
        if np.isnan(col_data).any():
            median = np.nanmedian(col_data)
            X[np.isnan(col_data), col_idx] = median
    return X, y, df


def train_model(X_train, y_train):
    """Entrena un Random Forest por clase con class_weight='balanced'."""
    base = RandomForestClassifier(**RF_PARAMS)
    model = MultiOutputClassifier(base, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def predict_proba_multilabel(model, X):
    """Devuelve un array (n_samples, n_labels) con la probabilidad de la clase positiva."""
    probas_per_label = model.predict_proba(X)  # lista de (n_samples, 2)
    probas = np.column_stack([p[:, 1] for p in probas_per_label])
    return probas


def evaluate_at_threshold(y_true, y_proba, threshold=0.5, label_names=None):
    """Métricas por clase y agregadas con un umbral dado."""
    y_pred = (y_proba >= threshold).astype(int)

    per_class = []
    for i, name in enumerate(label_names):
        per_class.append({
            "label": name,
            "support": int(y_true[:, i].sum()),
            "precision": precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "recall": recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "f1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "ap": average_precision_score(y_true[:, i], y_proba[:, i]),
        })

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    return pd.DataFrame(per_class), macro_f1, micro_f1, y_pred


def find_optimal_thresholds(y_true, y_proba, label_names):
    """Para cada clase, busca el umbral que maximiza F1 sobre las predicciones."""
    optimal = {}
    for i, name in enumerate(label_names):
        precs, recs, thresholds = precision_recall_curve(y_true[:, i], y_proba[:, i])
        # F1 = 2 * P * R / (P + R), evitar división por cero
        f1s = np.where(
            (precs + recs) > 0,
            2 * precs * recs / (precs + recs),
            0
        )
        # precision_recall_curve devuelve len(thresholds) = len(precs) - 1
        best_idx = np.argmax(f1s[:-1]) if len(f1s) > 1 else 0
        optimal[name] = {
            "threshold": float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5,
            "f1": float(f1s[best_idx]),
            "precision": float(precs[best_idx]),
            "recall": float(recs[best_idx]),
        }
    return optimal


def plot_confusion_matrices(y_true, y_pred, label_names, output_path):
    """Una matriz de confusión por clase (2x2 cada una)."""
    n = len(label_names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows))
    axes = axes.flatten()

    for i, name in enumerate(label_names):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
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


def plot_precision_recall_curves(y_true, y_proba, label_names, output_path):
    """Curva precision-recall para cada clase."""
    plt.figure(figsize=(10, 8))
    for i, name in enumerate(label_names):
        precs, recs, _ = precision_recall_curve(y_true[:, i], y_proba[:, i])
        ap = average_precision_score(y_true[:, i], y_proba[:, i])
        plt.plot(recs, precs, label=f"{name} (AP={ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curvas Precision-Recall por clase")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_feature_importance(model, feature_names, label_names, output_path):
    """Importancia de features por clase (Random Forest tiene esto built-in)."""
    importances = []
    for i, name in enumerate(label_names):
        clf = model.estimators_[i]
        importances.append(clf.feature_importances_)
    imp_df = pd.DataFrame(importances, index=label_names, columns=feature_names)

    plt.figure(figsize=(12, 6))
    sns.heatmap(imp_df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Importancia de features por clase (Random Forest)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    return imp_df


def print_metrics_table(metrics_df, macro_f1, micro_f1, name):
    """Imprime una tabla limpia de métricas."""
    print(f"\n{'='*60}")
    print(f"Métricas en {name}")
    print(f"{'='*60}")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n  Macro-F1: {macro_f1:.3f}")
    print(f"  Micro-F1: {micro_f1:.3f}")


def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    print("Cargando splits...")
    X_train, y_train, _ = load_split(TRAIN_CSV)
    X_val, y_val, _ = load_split(VAL_CSV)
    X_test, y_test, _ = load_split(TEST_CSV)
    print(f"  Train: {X_train.shape[0]} frames")
    print(f"  Val:   {X_val.shape[0]} frames")
    print(f"  Test:  {X_test.shape[0]} frames")
    print(f"  Features: {len(FEATURE_COLUMNS)}")
    print(f"  Clases: {len(LABEL_COLUMNS)}")

    print("\nEntrenando Random Forest multietiqueta...")
    model = train_model(X_train, y_train)
    print("Entrenado.")

    # Probabilidades
    print("\nGenerando predicciones...")
    proba_val = predict_proba_multilabel(model, X_val)
    proba_test = predict_proba_multilabel(model, X_test)

    # ===== Evaluación con umbral 0.5 (default) =====
    print("\n" + "#" * 60)
    print("# RESULTADOS CON UMBRAL POR DEFECTO (0.5)")
    print("#" * 60)

    val_metrics, val_macro, val_micro, val_pred = evaluate_at_threshold(
        y_val, proba_val, 0.5, LABEL_COLUMNS
    )
    print_metrics_table(val_metrics, val_macro, val_micro, "VALIDATION")

    test_metrics, test_macro, test_micro, test_pred = evaluate_at_threshold(
        y_test, proba_test, 0.5, LABEL_COLUMNS
    )
    print_metrics_table(test_metrics, test_macro, test_micro, "TEST")

    val_metrics.to_csv(out / "metrics_val_thr05.csv", index=False)
    test_metrics.to_csv(out / "metrics_test_thr05.csv", index=False)

    # ===== Búsqueda de umbrales óptimos por clase (sobre VAL) =====
    print("\n" + "#" * 60)
    print("# UMBRALES ÓPTIMOS POR CLASE (calibrados sobre VAL)")
    print("#" * 60)

    optimal = find_optimal_thresholds(y_val, proba_val, LABEL_COLUMNS)
    opt_df = pd.DataFrame([
        {"label": k, **v} for k, v in optimal.items()
    ])
    print("\n", opt_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    opt_df.to_csv(out / "optimal_thresholds.csv", index=False)

    # Aplicar umbrales óptimos a test
    thresholds = np.array([optimal[c]["threshold"] for c in LABEL_COLUMNS])
    test_pred_opt = (proba_test >= thresholds).astype(int)

    print("\n" + "#" * 60)
    print("# RESULTADOS EN TEST CON UMBRALES ÓPTIMOS POR CLASE")
    print("#" * 60)

    opt_rows = []
    for i, name in enumerate(LABEL_COLUMNS):
        opt_rows.append({
            "label": name,
            "threshold": thresholds[i],
            "support": int(y_test[:, i].sum()),
            "precision": precision_score(y_test[:, i], test_pred_opt[:, i], zero_division=0),
            "recall": recall_score(y_test[:, i], test_pred_opt[:, i], zero_division=0),
            "f1": f1_score(y_test[:, i], test_pred_opt[:, i], zero_division=0),
        })
    opt_results_df = pd.DataFrame(opt_rows)
    print("\n", opt_results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    macro_f1_opt = f1_score(y_test, test_pred_opt, average="macro", zero_division=0)
    micro_f1_opt = f1_score(y_test, test_pred_opt, average="micro", zero_division=0)
    print(f"\n  Macro-F1 (test, umbrales óptimos): {macro_f1_opt:.3f}")
    print(f"  Micro-F1 (test, umbrales óptimos): {micro_f1_opt:.3f}")

    opt_results_df.to_csv(out / "metrics_test_optimal_thr.csv", index=False)

    # ===== Visualizaciones =====
    print("\nGenerando visualizaciones...")
    plot_confusion_matrices(y_test, test_pred_opt, LABEL_COLUMNS,
                            out / "confusion_matrices_test.png")
    plot_precision_recall_curves(y_test, proba_test, LABEL_COLUMNS,
                                 out / "precision_recall_curves_test.png")
    imp_df = plot_feature_importance(model, FEATURE_COLUMNS, LABEL_COLUMNS,
                                     out / "feature_importance.png")
    imp_df.to_csv(out / "feature_importance.csv")

    # ===== Guardar modelo =====
    if SAVE_MODEL:
        model_path = out / "rf_baseline.joblib"
        joblib.dump({
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "label_columns": LABEL_COLUMNS,
            "optimal_thresholds": {k: v["threshold"] for k, v in optimal.items()},
        }, model_path)
        print(f"\nModelo guardado en {model_path}")

    print(f"\nResultados completos en: {out.absolute()}")


if __name__ == "__main__":
    main()