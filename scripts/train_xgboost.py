"""
Modelo XGBoost multietiqueta para clasificación de errores en peso muerto.

Mejoras sobre el baseline Random Forest:
  - XGBoost suele superar a Random Forest en datasets tabulares con features
    correlacionadas (vuestro caso con varias features de fase relacionadas).
  - Cada clase tiene su scale_pos_weight calculado a partir del desbalance
    (equivalente al class_weight='balanced' de RF).
  - Early stopping sobre validation: evita el sobreajuste en clases pequeñas
    como Distancia, parando cuando el modelo deja de mejorar en val.
  - Un modelo por clase (estrategia OneVsRest manual) para controlar mejor
    los hiperparámetros y reportes por clase.

Evalúa con umbral 0.5 y con umbrales óptimos calibrados sobre VAL.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
import joblib

# ---------- CONFIGURACIÓN ----------
TRAIN_CSV = "splits/dataset_train.csv"
VAL_CSV = "splits/dataset_val.csv"
TEST_CSV = "splits/dataset_test.csv"
OUTPUT_DIR = "resultados_xgboost"

LABEL_COLUMNS = ["Agarre", "Cabeza", "Distancia", "Dorsal", "Lumbar", "Pierna"]

FEATURE_COLUMNS = [
    "back_angle_vertical", "neck_angle",
    "knee_angle", "hip_angle", "grip_to_shoulder", "grip_to_hip",
    "bar_to_shin_x", "wrist_height_ratio", "avg_pose_conf", "side_detected",
    "knee_x_phase", "neck_x_phase", "knee_extension_low",
]

# Hiperparámetros base de XGBoost
XGB_PARAMS = {
    "n_estimators": 500,           # se cortará por early stopping
    "max_depth": 6,                # profundidad moderada para evitar overfitting
    "learning_rate": 0.05,         # tasa baja, compensada por early stopping
    "min_child_weight": 3,         # regularización: mínimo de "peso" en hojas
    "subsample": 0.8,              # 80% de filas por árbol (regularización)
    "colsample_bytree": 0.8,       # 80% de features por árbol
    "reg_alpha": 0.1,              # L1
    "reg_lambda": 1.0,             # L2
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "early_stopping_rounds": 30,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

SAVE_MODEL = True
# -----------------------------------


def load_split(path):
    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df[LABEL_COLUMNS].values.astype(np.float32)
    # Imputación con mediana para NaN residuales
    for col_idx in range(X.shape[1]):
        col_data = X[:, col_idx]
        if np.isnan(col_data).any():
            median = np.nanmedian(col_data)
            X[np.isnan(col_data), col_idx] = median
    return X, y, df


def train_per_class(X_train, y_train, X_val, y_val, label_names):
    """Entrena un XGBoost por clase con scale_pos_weight y early stopping."""
    models = {}
    best_iters = {}
    for i, name in enumerate(label_names):
        y_tr = y_train[:, i]
        y_vl = y_val[:, i]

        # scale_pos_weight = ratio de negativos / positivos en train
        n_pos = (y_tr == 1).sum()
        n_neg = (y_tr == 0).sum()
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        params = {**XGB_PARAMS, "scale_pos_weight": spw}
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_tr,
            eval_set=[(X_val, y_vl)],
            verbose=False,
        )
        models[name] = model
        best_iters[name] = model.best_iteration
        print(f"  {name}: scale_pos_weight={spw:.2f}, "
              f"best_iter={model.best_iteration}")
    return models, best_iters


def predict_proba_multilabel(models, X, label_names):
    """Devuelve array (n_samples, n_labels) con probas de clase positiva."""
    probas = np.zeros((X.shape[0], len(label_names)), dtype=np.float32)
    for i, name in enumerate(label_names):
        probas[:, i] = models[name].predict_proba(X)[:, 1]
    return probas


def evaluate_at_threshold(y_true, y_proba, threshold, label_names):
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
    optimal = {}
    for i, name in enumerate(label_names):
        precs, recs, thresholds = precision_recall_curve(y_true[:, i], y_proba[:, i])
        # F1 = 2 * P * R / (P + R), evitando NaN cuando P + R = 0
        denom = precs + recs
        with np.errstate(divide="ignore", invalid="ignore"):
            f1s = np.where(denom > 0, 2 * precs * recs / denom, 0)
        best_idx = np.argmax(f1s[:-1]) if len(f1s) > 1 else 0
        optimal[name] = {
            "threshold": float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5,
            "f1": float(f1s[best_idx]),
            "precision": float(precs[best_idx]),
            "recall": float(recs[best_idx]),
        }
    return optimal


def plot_confusion_matrices(y_true, y_pred, label_names, output_path):
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
    plt.figure(figsize=(10, 8))
    for i, name in enumerate(label_names):
        precs, recs, _ = precision_recall_curve(y_true[:, i], y_proba[:, i])
        ap = average_precision_score(y_true[:, i], y_proba[:, i])
        plt.plot(recs, precs, label=f"{name} (AP={ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curvas Precision-Recall por clase (XGBoost)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_feature_importance(models, feature_names, label_names, output_path):
    importances = []
    for name in label_names:
        clf = models[name]
        # XGBoost: usar gain como medida (importancia por contribución a la pérdida)
        booster = clf.get_booster()
        score = booster.get_score(importance_type="gain")
        # score es dict {f0: ..., f1: ...} -> mapear a features
        imp_vec = []
        for i in range(len(feature_names)):
            imp_vec.append(score.get(f"f{i}", 0.0))
        # Normalizar para que sumen 1 (comparable con RF)
        total = sum(imp_vec)
        imp_vec = [v / total if total > 0 else 0 for v in imp_vec]
        importances.append(imp_vec)

    imp_df = pd.DataFrame(importances, index=label_names, columns=feature_names)

    plt.figure(figsize=(14, 6))
    sns.heatmap(imp_df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Importancia de features por clase (XGBoost, normalized gain)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    return imp_df


def print_metrics_table(metrics_df, macro_f1, micro_f1, name):
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

    print("\nEntrenando XGBoost por clase con early stopping en VAL...")
    models, best_iters = train_per_class(X_train, y_train, X_val, y_val, LABEL_COLUMNS)

    print("\nGenerando predicciones...")
    proba_val = predict_proba_multilabel(models, X_val, LABEL_COLUMNS)
    proba_test = predict_proba_multilabel(models, X_test, LABEL_COLUMNS)

    # ===== Evaluación con umbral 0.5 =====
    print("\n" + "#" * 60)
    print("# RESULTADOS CON UMBRAL POR DEFECTO (0.5)")
    print("#" * 60)

    val_metrics, val_macro, val_micro, _ = evaluate_at_threshold(
        y_val, proba_val, 0.5, LABEL_COLUMNS)
    print_metrics_table(val_metrics, val_macro, val_micro, "VALIDATION")

    test_metrics, test_macro, test_micro, test_pred = evaluate_at_threshold(
        y_test, proba_test, 0.5, LABEL_COLUMNS)
    print_metrics_table(test_metrics, test_macro, test_micro, "TEST")

    val_metrics.to_csv(out / "metrics_val_thr05.csv", index=False)
    test_metrics.to_csv(out / "metrics_test_thr05.csv", index=False)

    # ===== Umbrales óptimos =====
    print("\n" + "#" * 60)
    print("# UMBRALES ÓPTIMOS POR CLASE (calibrados sobre VAL)")
    print("#" * 60)

    optimal = find_optimal_thresholds(y_val, proba_val, LABEL_COLUMNS)
    opt_df = pd.DataFrame([{"label": k, **v} for k, v in optimal.items()])
    print("\n", opt_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    opt_df.to_csv(out / "optimal_thresholds.csv", index=False)

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
    imp_df = plot_feature_importance(models, FEATURE_COLUMNS, LABEL_COLUMNS,
                                     out / "feature_importance.png")
    imp_df.to_csv(out / "feature_importance.csv")

    # ===== Guardar modelos =====
    if SAVE_MODEL:
        model_path = out / "xgboost_models.joblib"
        joblib.dump({
            "models": models,
            "feature_columns": FEATURE_COLUMNS,
            "label_columns": LABEL_COLUMNS,
            "optimal_thresholds": {k: v["threshold"] for k, v in optimal.items()},
            "best_iters": best_iters,
        }, model_path)
        print(f"\nModelos guardados en {model_path}")

    print(f"\nResultados completos en: {out.absolute()}")


if __name__ == "__main__":
    main()