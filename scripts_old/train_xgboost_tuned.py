"""
Grid search de hiperparámetros para XGBoost multietiqueta.

Hace una búsqueda independiente por clase, lo cual permite que cada modelo
binario tenga sus propios hiperparámetros óptimos. Esto suele dar mejor
resultado que usar los mismos para todas, especialmente cuando las clases
tienen propiedades muy distintas (Distancia rara, Cabeza balanceada, etc.).

Estrategia:
  1. Espacio de búsqueda compacto pero significativo.
  2. Búsqueda por random search (más eficiente que grid completo).
  3. Para cada combinación, evaluamos F1 sobre VAL con early stopping.
  4. Guardamos los mejores hiperparámetros por clase y reentrenamos el modelo
     final con esos parámetros.
  5. Comparación final con el XGBoost no tuneado.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import ParameterSampler
import joblib

# ---------- CONFIGURACIÓN ----------
TRAIN_CSV = "splits/dataset_train.csv"
VAL_CSV = "splits/dataset_val.csv"
TEST_CSV = "splits/dataset_test.csv"
OUTPUT_DIR = "resultados_xgboost_tuneado"

LABEL_COLUMNS = ["Agarre", "Cabeza", "Distancia", "Dorsal", "Lumbar", "Pierna"]
FEATURE_COLUMNS = [
    "back_angle_vertical", "neck_angle",
    "knee_angle", "hip_angle", "grip_to_shoulder", "grip_to_hip",
    "bar_to_shin_x", "wrist_height_ratio", "avg_pose_conf", "side_detected",
    "knee_x_phase", "neck_x_phase", "knee_extension_low",
]

# Espacio de búsqueda
PARAM_SPACE = {
    "max_depth": [4, 5, 6, 7, 8],
    "learning_rate": [0.03, 0.05, 0.08, 0.1],
    "min_child_weight": [1, 3, 5, 8],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0],
}

# Parámetros fijos
FIXED_PARAMS = {
    "n_estimators": 1000,           # ya cortado por early stopping
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "early_stopping_rounds": 30,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

N_TRIALS_PER_CLASS = 40    # combinaciones por clase
RANDOM_SEED = 42
# -----------------------------------


def load_split(path):
    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df[LABEL_COLUMNS].values.astype(np.float32)
    for col_idx in range(X.shape[1]):
        col_data = X[:, col_idx]
        if np.isnan(col_data).any():
            median = np.nanmedian(col_data)
            X[np.isnan(col_data), col_idx] = median
    return X, y, df


def best_f1_threshold(y_true, y_proba):
    """Mejor F1 alcanzable variando el umbral, y el umbral correspondiente."""
    precs, recs, thresholds = precision_recall_curve(y_true, y_proba)
    denom = precs + recs
    with np.errstate(divide="ignore", invalid="ignore"):
        f1s = np.where(denom > 0, 2 * precs * recs / denom, 0)
    if len(thresholds) == 0:
        return 0.0, 0.5
    best_idx = np.argmax(f1s[:-1])
    return float(f1s[best_idx]), float(thresholds[best_idx])


def tune_class(X_train, y_train_col, X_val, y_val_col,
               param_space, n_trials, fixed_params, scale_pos_weight, seed):
    """Random search para una clase. Devuelve mejores params, mejor F1, mejor umbral."""
    sampler = ParameterSampler(param_space, n_iter=n_trials, random_state=seed)
    best_score = -1
    best_params = None
    best_threshold = 0.5

    for trial_idx, params in enumerate(sampler):
        all_params = {**fixed_params, **params, "scale_pos_weight": scale_pos_weight}
        model = XGBClassifier(**all_params)
        try:
            model.fit(
                X_train, y_train_col,
                eval_set=[(X_val, y_val_col)],
                verbose=False,
            )
        except Exception as e:
            print(f"    Trial {trial_idx} falló: {e}")
            continue

        proba = model.predict_proba(X_val)[:, 1]
        f1, thr = best_f1_threshold(y_val_col, proba)

        if f1 > best_score:
            best_score = f1
            best_params = params
            best_threshold = thr

    return best_params, best_score, best_threshold


def train_final_model(X_train, y_train_col, X_val, y_val_col,
                      best_params, fixed_params, scale_pos_weight):
    """Entrena el modelo final con los mejores hiperparámetros."""
    all_params = {**fixed_params, **best_params, "scale_pos_weight": scale_pos_weight}
    model = XGBClassifier(**all_params)
    model.fit(
        X_train, y_train_col,
        eval_set=[(X_val, y_val_col)],
        verbose=False,
    )
    return model


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


def evaluate_with_per_class_thresholds(y_true, y_proba, thresholds, label_names):
    """Aplica un umbral distinto a cada clase."""
    y_pred = np.zeros_like(y_true, dtype=int)
    for i in range(len(label_names)):
        y_pred[:, i] = (y_proba[:, i] >= thresholds[i]).astype(int)
    per_class = []
    for i, name in enumerate(label_names):
        per_class.append({
            "label": name,
            "threshold": thresholds[i],
            "support": int(y_true[:, i].sum()),
            "precision": precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "recall": recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "f1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
        })
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    return pd.DataFrame(per_class), macro_f1, micro_f1, y_pred


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
    print(f"\nBúsqueda: {N_TRIALS_PER_CLASS} trials por clase ({N_TRIALS_PER_CLASS * len(LABEL_COLUMNS)} en total)")

    best_params_per_class = {}
    best_val_f1_per_class = {}
    best_thresholds_per_class = {}
    final_models = {}

    for class_idx, class_name in enumerate(LABEL_COLUMNS):
        y_tr = y_train[:, class_idx]
        y_vl = y_val[:, class_idx]

        n_pos = (y_tr == 1).sum()
        n_neg = (y_tr == 0).sum()
        spw = n_neg / max(n_pos, 1)

        print(f"\n{'='*60}")
        print(f"Tuneando '{class_name}' (scale_pos_weight={spw:.2f})")
        print(f"{'='*60}")

        best_params, best_f1, best_thr = tune_class(
            X_train, y_tr, X_val, y_vl,
            PARAM_SPACE, N_TRIALS_PER_CLASS, FIXED_PARAMS, spw,
            seed=RANDOM_SEED + class_idx,
        )

        print(f"  Mejor F1 en val: {best_f1:.4f} (umbral {best_thr:.3f})")
        print(f"  Mejores params: {best_params}")

        # Reentrenar modelo final
        model = train_final_model(X_train, y_tr, X_val, y_vl,
                                  best_params, FIXED_PARAMS, spw)

        best_params_per_class[class_name] = best_params
        best_val_f1_per_class[class_name] = best_f1
        best_thresholds_per_class[class_name] = best_thr
        final_models[class_name] = model

    # ===== Evaluación final =====
    print("\n" + "#" * 60)
    print("# EVALUACIÓN FINAL DEL MODELO TUNEADO")
    print("#" * 60)

    proba_test = np.column_stack([
        final_models[c].predict_proba(X_test)[:, 1] for c in LABEL_COLUMNS
    ])

    # 1) Con umbral 0.5
    test_metrics_05, macro_05, micro_05, pred_05 = evaluate_at_threshold(
        y_test, proba_test, 0.5, LABEL_COLUMNS
    )
    print("\n--- TEST con umbral 0.5 ---")
    print(test_metrics_05.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n  Macro-F1: {macro_05:.3f}, Micro-F1: {micro_05:.3f}")
    test_metrics_05.to_csv(out / "metrics_test_thr05.csv", index=False)

    # 2) Con umbrales óptimos por clase (de la búsqueda)
    thresholds_arr = np.array([best_thresholds_per_class[c] for c in LABEL_COLUMNS])
    test_metrics_opt, macro_opt, micro_opt, pred_opt = evaluate_with_per_class_thresholds(
        y_test, proba_test, thresholds_arr, LABEL_COLUMNS
    )
    print("\n--- TEST con umbrales óptimos por clase ---")
    print(test_metrics_opt.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n  Macro-F1: {macro_opt:.3f}, Micro-F1: {micro_opt:.3f}")
    test_metrics_opt.to_csv(out / "metrics_test_optimal_thr.csv", index=False)

    # Resumen de hiperparámetros
    print("\n" + "#" * 60)
    print("# MEJORES HIPERPARÁMETROS POR CLASE")
    print("#" * 60)
    params_df = pd.DataFrame([
        {"label": c, **best_params_per_class[c],
         "best_val_f1": best_val_f1_per_class[c],
         "threshold": best_thresholds_per_class[c]}
        for c in LABEL_COLUMNS
    ])
    print(params_df.to_string(index=False))
    params_df.to_csv(out / "best_hyperparameters.csv", index=False)

    # Visualizaciones
    plot_confusion_matrices(y_test, pred_opt, LABEL_COLUMNS,
                            out / "confusion_matrices_test.png")

    # Guardar modelos
    joblib.dump({
        "models": final_models,
        "feature_columns": FEATURE_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "best_thresholds": best_thresholds_per_class,
        "best_params": best_params_per_class,
    }, out / "xgboost_tuned.joblib")

    print(f"\nResultados completos en: {out.absolute()}")


if __name__ == "__main__":
    main()