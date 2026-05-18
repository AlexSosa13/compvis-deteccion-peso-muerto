"""
MLSMOTE conservador: variante más cuidadosa del oversampling multietiqueta.

Cambios respecto a la versión anterior:
  - K=3 vecinos (antes K=5): reduce el "arrastre" de etiquetas correlacionadas.
  - Menos sintéticas: 200 para Distancia (antes 400) y 0 para Agarre (ya no
    es realmente minoritaria con 310 ejemplos).
  - LABEL_PROPAGATION_MODE controla cómo se asignan etiquetas a las sintéticas:
      'minority_only': la sintética conserva SOLO la clase minoritaria activa.
                       Resto de etiquetas a 0. Evita el problema visto antes
                       donde Dorsal y Pierna ganaban positivos espurios.
      'voting': la versión anterior (votación de los k+1 vecinos).
      'seed_copy': hereda exactamente las etiquetas de la semilla.

Solo se aplica al split de TRAIN.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# ---------- CONFIGURACIÓN ----------
INPUT_TRAIN_CSV = "splits/dataset_train.csv"
OUTPUT_CSV = "splits/dataset_train_augmented.csv"

LABEL_COLUMNS = ["Agarre", "Cabeza", "Distancia", "Dorsal", "Lumbar", "Pierna"]

FEATURE_COLUMNS = [
    "back_angle_vertical", "neck_angle",
    "knee_angle", "hip_angle", "grip_to_shoulder", "grip_to_hip",
    "bar_to_shin_x", "wrist_height_ratio", "avg_pose_conf", "side_detected",
    "knee_x_phase", "neck_x_phase", "knee_extension_low",
]

# Sólo aumentamos Distancia, que es la única realmente minoritaria
AUGMENTATION_TARGETS = {
    "Distancia": 200,   # actualmente 122, queremos casi duplicar
}

# Variante conservadora
K_NEIGHBORS = 3
LABEL_PROPAGATION_MODE = "minority_only"  # 'minority_only' | 'voting' | 'seed_copy'

# Normalización de features antes de buscar vecinos (importante para
# evitar que features con rangos grandes dominen la distancia)
NORMALIZE_FOR_KNN = True

RANDOM_SEED = 42
# -----------------------------------


def normalize_for_knn(X):
    """Z-score normalization para que la distancia euclídea sea más razonable."""
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds = np.where(stds > 1e-9, stds, 1.0)  # evitar dividir por cero
    return (X - means) / stds, means, stds


def get_minority_samples(y, label_idx):
    return np.where(y[:, label_idx] == 1)[0]


def mlsmote_generate(X_full, y_full, minority_indices, n_to_generate, k_neighbors,
                     rng, label_idx, propagation_mode):
    """Genera n_to_generate muestras sintéticas para una clase minoritaria.

    Para encontrar vecinos, usa SOLO las muestras donde la clase minoritaria
    está activa (asegura que las sintéticas caigan en el subespacio de la clase).
    """
    if len(minority_indices) < k_neighbors + 1:
        print(f"  Aviso: solo hay {len(minority_indices)} muestras minoritarias, "
              f"insuficientes para k={k_neighbors}. Saltando.")
        return np.empty((0, X_full.shape[1])), np.empty((0, y_full.shape[1]))

    X_minority = X_full[minority_indices]
    y_minority = y_full[minority_indices]

    # KNN sobre el subespacio de la clase minoritaria
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
    nn.fit(X_minority)
    _, neighbor_idx = nn.kneighbors(X_minority)

    synthetic_X = []
    synthetic_y = []

    for _ in range(n_to_generate):
        seed_local_idx = rng.integers(0, len(minority_indices))
        seed_x = X_minority[seed_local_idx]
        seed_y = y_minority[seed_local_idx]

        # Excluir la propia muestra (índice 0) de los vecinos
        neighbors_local = neighbor_idx[seed_local_idx][1:]
        chosen_neighbor_local = rng.choice(neighbors_local)
        neighbor_x = X_minority[chosen_neighbor_local]

        # Interpolar features
        lam = rng.uniform(0, 1)
        new_x = seed_x + lam * (neighbor_x - seed_x)

        # Asignar etiquetas según el modo
        if propagation_mode == "minority_only":
            # Solo la clase minoritaria objetivo activa
            new_y = np.zeros(y_full.shape[1], dtype=int)
            new_y[label_idx] = 1
        elif propagation_mode == "seed_copy":
            # Hereda exactamente las etiquetas de la semilla
            new_y = seed_y.astype(int).copy()
        elif propagation_mode == "voting":
            all_labels = y_minority[np.append(neighbors_local, seed_local_idx)]
            votes = all_labels.sum(axis=0)
            threshold = (k_neighbors + 1) / 2
            new_y = (votes > threshold).astype(int)
            if new_y.sum() == 0:
                new_y = seed_y.astype(int)
        else:
            raise ValueError(f"Modo desconocido: {propagation_mode}")

        synthetic_X.append(new_x)
        synthetic_y.append(new_y)

    return np.array(synthetic_X), np.array(synthetic_y)


def main():
    print(f"Cargando {INPUT_TRAIN_CSV}...")
    df = pd.read_csv(INPUT_TRAIN_CSV)
    print(f"Train original: {len(df)} filas")
    print(f"Modo de propagación de etiquetas: {LABEL_PROPAGATION_MODE}")
    print(f"K vecinos: {K_NEIGHBORS}")

    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    y = df[LABEL_COLUMNS].values.astype(np.int32)

    # Imputación de NaN residuales
    for col_idx in range(X.shape[1]):
        col_data = X[:, col_idx]
        if np.isnan(col_data).any():
            median = np.nanmedian(col_data)
            X[np.isnan(col_data), col_idx] = median

    # Normalización para KNN (la usamos solo para buscar vecinos, no para guardar)
    if NORMALIZE_FOR_KNN:
        X_norm, _, _ = normalize_for_knn(X)
    else:
        X_norm = X.copy()

    print("\nDistribución original:")
    for i, name in enumerate(LABEL_COLUMNS):
        n = int(y[:, i].sum())
        print(f"  {name}: {n} positivos")

    rng = np.random.default_rng(RANDOM_SEED)

    all_synth_X = []
    all_synth_y = []

    for label_name, target in AUGMENTATION_TARGETS.items():
        if label_name not in LABEL_COLUMNS:
            continue
        label_idx = LABEL_COLUMNS.index(label_name)
        current = int(y[:, label_idx].sum())
        n_to_gen = max(0, target - current)
        if n_to_gen == 0:
            print(f"\n  {label_name}: ya tiene {current} >= {target}, no se genera.")
            continue

        print(f"\nGenerando {n_to_gen} muestras sintéticas para '{label_name}' "
              f"(actual: {current}, objetivo: {target})...")
        minority_indices = get_minority_samples(y, label_idx)

        # Usamos X_norm para buscar vecinos pero generamos la sintética en el
        # espacio original interpolando con los valores reales.
        # Necesitamos los vecinos por índice y luego mezclar en X original.
        nn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1)
        nn.fit(X_norm[minority_indices])
        _, neighbor_idx_arr = nn.kneighbors(X_norm[minority_indices])

        synth_X = []
        synth_y = []
        for _ in range(n_to_gen):
            seed_local_idx = rng.integers(0, len(minority_indices))
            seed_x = X[minority_indices[seed_local_idx]]
            seed_y = y[minority_indices[seed_local_idx]]

            neighbors_local = neighbor_idx_arr[seed_local_idx][1:]
            chosen_local = rng.choice(neighbors_local)
            neighbor_x = X[minority_indices[chosen_local]]

            lam = rng.uniform(0, 1)
            new_x = seed_x + lam * (neighbor_x - seed_x)

            if LABEL_PROPAGATION_MODE == "minority_only":
                new_y = np.zeros(y.shape[1], dtype=int)
                new_y[label_idx] = 1
            elif LABEL_PROPAGATION_MODE == "seed_copy":
                new_y = seed_y.astype(int).copy()
            elif LABEL_PROPAGATION_MODE == "voting":
                neighbors_y = y[minority_indices[np.append(neighbors_local, seed_local_idx)]]
                votes = neighbors_y.sum(axis=0)
                threshold = (K_NEIGHBORS + 1) / 2
                new_y = (votes > threshold).astype(int)
                if new_y.sum() == 0:
                    new_y = seed_y.astype(int)
            else:
                raise ValueError(f"Modo desconocido: {LABEL_PROPAGATION_MODE}")

            synth_X.append(new_x)
            synth_y.append(new_y)

        synth_X = np.array(synth_X)
        synth_y = np.array(synth_y)
        all_synth_X.append(synth_X)
        all_synth_y.append(synth_y)

        # Distribución de etiquetas en las sintéticas generadas
        synth_dist = {LABEL_COLUMNS[j]: int(synth_y[:, j].sum())
                      for j in range(len(LABEL_COLUMNS))}
        print(f"  Generadas {len(synth_X)} muestras.")
        print(f"  Distribución sintética: {synth_dist}")

    if all_synth_X:
        synth_X_full = np.vstack(all_synth_X)
        synth_y_full = np.vstack(all_synth_y)

        X_aug = np.vstack([X, synth_X_full])
        y_aug = np.vstack([y, synth_y_full])

        out_df = pd.DataFrame(X_aug, columns=FEATURE_COLUMNS)
        for i, name in enumerate(LABEL_COLUMNS):
            out_df[name] = y_aug[:, i]

        out_df["is_synthetic"] = ([0] * len(X)) + ([1] * len(synth_X_full))
        original_filenames = df["filename"].tolist()
        synthetic_filenames = [f"synth_{i:05d}.jpg" for i in range(len(synth_X_full))]
        out_df["filename"] = original_filenames + synthetic_filenames

        if "pose_valid" in df.columns:
            out_df["pose_valid"] = list(df["pose_valid"]) + [1] * len(synth_X_full)

        cols_order = ["filename"] + FEATURE_COLUMNS + LABEL_COLUMNS + ["is_synthetic"]
        if "pose_valid" in out_df.columns:
            cols_order.insert(-1, "pose_valid")
        out_df = out_df[cols_order]
    else:
        print("\nNo se generaron muestras sintéticas. Copiando train original.")
        out_df = df.copy()
        out_df["is_synthetic"] = 0

    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*50}")
    print(f"RESUMEN")
    print(f"{'='*50}")
    print(f"Train original:    {len(df)} filas")
    print(f"Train aumentado:   {len(out_df)} filas")
    print(f"Sintéticas:        {len(out_df) - len(df)}")
    print(f"\nDistribución final:")
    for name in LABEL_COLUMNS:
        n_total = int(out_df[name].sum())
        n_synth = int(out_df[out_df["is_synthetic"] == 1][name].sum())
        print(f"  {name}: {n_total} total ({n_synth} sintéticas)")

    print(f"\nGuardado en {OUTPUT_CSV}")


if __name__ == "__main__":
    main()