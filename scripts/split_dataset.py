"""
Genera el split train/validation/test para el dataset de features.

Estrategia: split aleatorio estratificado a nivel de frame, con semilla fija.
NOTA: No se hace split por vídeo ni por sujeto. Esta limitación debe documentarse
en la memoria del proyecto.

Estratificación: usamos 'iterative-stratification' para problemas multietiqueta,
que garantiza que la distribución de cada etiqueta se mantenga similar entre
los splits, incluyendo las clases minoritarias.

Si no tenéis instalada la librería:
    pip install iterative-stratification

Salida:
  - dataset_train.csv, dataset_val.csv, dataset_test.csv
  - Resumen por consola con la distribución de etiquetas en cada split.
"""

import pandas as pd
import numpy as np

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    HAS_ITERSTRAT = True
except ImportError:
    HAS_ITERSTRAT = False
    print("Aviso: 'iterative-stratification' no instalada.")
    print("Para mejor estratificación multietiqueta:")
    print("  pip install iterative-stratification")
    print("De momento se hará un split aleatorio simple.\n")

# ---------- CONFIGURACIÓN ----------
INPUT_CSV = "output/dataset_features.csv"
TRAIN_CSV = "splits/dataset_train.csv"
VAL_CSV = "splits/dataset_val.csv"
TEST_CSV = "splits/dataset_test.csv"

# Proporciones (deben sumar 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

LABEL_COLUMNS = ["Agarre", "Cabeza", "Distancia", "Dorsal", "Lumbar", "Pierna"]

# Filtrar frames sin pose válida antes de splitear (recomendado)
FILTER_INVALID_POSE = True
# -----------------------------------


def stratified_split(df, label_cols, train_ratio, val_ratio, test_ratio, seed):
    """Split estratificado multietiqueta usando iterative-stratification."""
    X = df.index.values.reshape(-1, 1)
    y = df[label_cols].values

    # Primero separamos test del resto
    test_frac = test_ratio
    splitter_test = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_frac, random_state=seed
    )
    trainval_idx, test_idx = next(splitter_test.split(X, y))

    # Luego separamos val de train (recalculando proporción sobre el resto)
    val_frac_of_remaining = val_ratio / (train_ratio + val_ratio)
    splitter_val = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_frac_of_remaining, random_state=seed
    )
    X_remaining = X[trainval_idx]
    y_remaining = y[trainval_idx]
    train_rel_idx, val_rel_idx = next(splitter_val.split(X_remaining, y_remaining))

    train_idx = trainval_idx[train_rel_idx]
    val_idx = trainval_idx[val_rel_idx]

    return train_idx, val_idx, test_idx


def random_split(df, train_ratio, val_ratio, test_ratio, seed):
    """Split aleatorio simple (fallback si no hay iterative-stratification)."""
    n = len(df)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


def print_label_distribution(df, name, label_cols):
    """Muestra la distribución de etiquetas y el conteo de 'Bien' (todas a 0)."""
    print(f"\n{name} ({len(df)} frames):")
    for col in label_cols:
        count = int(df[col].sum())
        pct = 100 * count / len(df)
        print(f"  {col}: {count} ({pct:.1f}%)")
    bien_count = int((df[label_cols].sum(axis=1) == 0).sum())
    pct = 100 * bien_count / len(df)
    print(f"  Bien (implícito): {bien_count} ({pct:.1f}%)")


def main():
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
        "Las proporciones deben sumar 1"

    df = pd.read_csv(INPUT_CSV)
    print(f"Dataset inicial: {len(df)} filas")

    if FILTER_INVALID_POSE and "pose_valid" in df.columns:
        before = len(df)
        df = df[df["pose_valid"] == 1].reset_index(drop=True)
        print(f"Tras filtrar poses inválidas: {len(df)} filas (eliminadas {before - len(df)})")

    # Verificar que tenemos las columnas de etiquetas
    missing = [c for c in LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas de etiqueta no encontradas: {missing}")

    # Split
    if HAS_ITERSTRAT:
        print("\nAplicando split estratificado multietiqueta...")
        train_idx, val_idx, test_idx = stratified_split(
            df, LABEL_COLUMNS, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
        )
    else:
        print("\nAplicando split aleatorio simple...")
        train_idx, val_idx, test_idx = random_split(
            df, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
        )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Guardar
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    # Resumen
    print(f"\n{'='*50}")
    print(f"RESUMEN")
    print(f"{'='*50}")
    print(f"Total: {len(df)} frames")
    print(f"  Train: {len(train_df)} ({100*len(train_df)/len(df):.1f}%)")
    print(f"  Val:   {len(val_df)} ({100*len(val_df)/len(df):.1f}%)")
    print(f"  Test:  {len(test_df)} ({100*len(test_df)/len(df):.1f}%)")

    print_label_distribution(train_df, "TRAIN", LABEL_COLUMNS)
    print_label_distribution(val_df, "VAL", LABEL_COLUMNS)
    print_label_distribution(test_df, "TEST", LABEL_COLUMNS)

    print(f"\nArchivos guardados: {TRAIN_CSV}, {VAL_CSV}, {TEST_CSV}")


if __name__ == "__main__":
    main()