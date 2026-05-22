"""
Análisis exploratorio del dataset de features.

Genera:
  - Resumen estadístico por feature
  - Histogramas de cada feature, separados por clase (frame con esa etiqueta vs sin ella)
  - Matriz de correlación entre features
  - Boxplots por clase para identificar features discriminativas
  - Resumen de combinaciones de etiquetas (qué errores aparecen juntos)

Salida: PNGs y CSVs en una carpeta de análisis.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- CONFIGURACIÓN ----------
INPUT_CSV = "output/dataset_features.csv"
OUTPUT_DIR = "output/analisis_exploratorio"

LABEL_COLUMNS = ["Agarre", "Cabeza", "Distancia", "Dorsal", "Lumbar", "Pierna"]

FEATURE_COLUMNS = [
    "back_angle_vertical", "neck_angle",
    "knee_angle", "hip_angle", "grip_to_shoulder", "grip_to_hip",
    "bar_to_shin_x", "wrist_height_ratio", "avg_pose_conf", "side_detected",
    "knee_x_phase", "neck_x_phase", "knee_extension_low",
]

# Filtrar frames sin pose válida para el análisis
FILTER_INVALID_POSE = True
# -----------------------------------


def setup_output(output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "histograms").mkdir(exist_ok=True)
    (out / "boxplots").mkdir(exist_ok=True)
    return out


def summary_statistics(df, feature_cols, output_dir):
    """Estadísticos descriptivos de cada feature."""
    stats = df[feature_cols].describe().T
    stats["n_nan"] = df[feature_cols].isna().sum()
    stats["pct_nan"] = 100 * stats["n_nan"] / len(df)
    stats.to_csv(output_dir / "feature_statistics.csv")
    print("\nResumen estadístico:")
    print(stats[["mean", "std", "min", "max", "pct_nan"]].round(2))


def label_distribution(df, label_cols, output_dir):
    """Distribución de etiquetas y combinaciones."""
    print("\nDistribución de etiquetas:")
    counts = []
    for col in label_cols:
        n_pos = int(df[col].sum())
        pct = 100 * n_pos / len(df)
        counts.append({"label": col, "count": n_pos, "pct": pct})
        print(f"  {col}: {n_pos} ({pct:.1f}%)")

    # 'Bien' implícito
    bien = int((df[label_cols].sum(axis=1) == 0).sum())
    counts.append({"label": "Bien (implícito)", "count": bien, "pct": 100 * bien / len(df)})
    print(f"  Bien (implícito): {bien} ({100*bien/len(df):.1f}%)")

    pd.DataFrame(counts).to_csv(output_dir / "label_distribution.csv", index=False)

    # Distribución del número de errores por frame
    n_errors = df[label_cols].sum(axis=1)
    print(f"\nNúmero de errores simultáneos por frame:")
    for k in range(int(n_errors.max()) + 1):
        n = int((n_errors == k).sum())
        print(f"  {k} errores: {n} frames ({100*n/len(df):.1f}%)")

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x="label", y="count",
                data=pd.DataFrame(counts), ax=ax[0])
    ax[0].set_title("Distribución de etiquetas")
    ax[0].tick_params(axis='x', rotation=45)

    n_errors.value_counts().sort_index().plot(kind="bar", ax=ax[1])
    ax[1].set_title("Número de errores simultáneos por frame")
    ax[1].set_xlabel("Número de errores")
    ax[1].set_ylabel("Frames")

    plt.tight_layout()
    plt.savefig(output_dir / "label_distribution.png", dpi=120)
    plt.close()


def label_cooccurrence(df, label_cols, output_dir):
    """Matriz de co-ocurrencia entre etiquetas."""
    cooc = np.zeros((len(label_cols), len(label_cols)), dtype=int)
    for i, a in enumerate(label_cols):
        for j, b in enumerate(label_cols):
            cooc[i, j] = int(((df[a] == 1) & (df[b] == 1)).sum())

    cooc_df = pd.DataFrame(cooc, index=label_cols, columns=label_cols)
    cooc_df.to_csv(output_dir / "label_cooccurrence.csv")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cooc_df, annot=True, fmt="d", cmap="YlOrRd", cbar=True)
    plt.title("Co-ocurrencia de etiquetas (diagonal = total por clase)")
    plt.tight_layout()
    plt.savefig(output_dir / "label_cooccurrence.png", dpi=120)
    plt.close()


def feature_correlation(df, feature_cols, output_dir):
    """Matriz de correlación entre features."""
    corr = df[feature_cols].corr()
    corr.to_csv(output_dir / "feature_correlation.csv")

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, vmin=-1, vmax=1)
    plt.title("Correlación entre features")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_correlation.png", dpi=120)
    plt.close()


def histograms_by_class(df, feature_cols, label_cols, output_dir):
    """Histograma de cada feature, separando frames con vs sin cada etiqueta."""
    hist_dir = output_dir / "histograms"
    for label in label_cols:
        n_feats = len(feature_cols)
        ncols = 3
        nrows = (n_feats + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
        axes = axes.flatten()

        for i, feat in enumerate(feature_cols):
            ax = axes[i]
            data_pos = df[df[label] == 1][feat].dropna()
            data_neg = df[df[label] == 0][feat].dropna()

            if len(data_pos) == 0 or len(data_neg) == 0:
                ax.set_visible(False)
                continue

            ax.hist(data_neg, bins=30, alpha=0.5, label=f"{label}=0", density=True)
            ax.hist(data_pos, bins=30, alpha=0.5, label=f"{label}=1", density=True)
            ax.set_title(feat)
            ax.legend(fontsize=8)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f"Distribución de features para etiqueta: {label}", fontsize=14)
        plt.tight_layout()
        plt.savefig(hist_dir / f"hist_{label}.png", dpi=100)
        plt.close()
    print(f"\nHistogramas guardados en {hist_dir}/")


def boxplots_by_class(df, feature_cols, label_cols, output_dir):
    """Boxplots de cada feature por clase."""
    box_dir = output_dir / "boxplots"
    for feat in feature_cols:
        fig, axes = plt.subplots(1, len(label_cols), figsize=(20, 4), sharey=True)
        for i, label in enumerate(label_cols):
            data = df[[feat, label]].dropna()
            if len(data) == 0:
                continue
            sns.boxplot(x=label, y=feat, data=data, ax=axes[i])
            axes[i].set_title(label)
            axes[i].set_xlabel("")
        plt.suptitle(f"Feature: {feat}", fontsize=14)
        plt.tight_layout()
        plt.savefig(box_dir / f"box_{feat}.png", dpi=100)
        plt.close()
    print(f"Boxplots guardados en {box_dir}/")


def feature_discrimination_scores(df, feature_cols, label_cols, output_dir):
    """Calcula un score simple de cuánto discrimina cada feature cada clase.
    Usa la diferencia normalizada entre medias (efecto Cohen's d aproximado).
    """
    rows = []
    for label in label_cols:
        for feat in feature_cols:
            pos = df[df[label] == 1][feat].dropna()
            neg = df[df[label] == 0][feat].dropna()
            if len(pos) < 2 or len(neg) < 2:
                d = np.nan
            else:
                pooled_std = np.sqrt((pos.var() + neg.var()) / 2)
                if pooled_std == 0:
                    d = np.nan
                else:
                    d = abs(pos.mean() - neg.mean()) / pooled_std
            rows.append({"label": label, "feature": feat, "discrimination": d})

    scores = pd.DataFrame(rows)
    scores_pivot = scores.pivot(index="feature", columns="label", values="discrimination")
    scores_pivot.to_csv(output_dir / "feature_discrimination.csv")

    plt.figure(figsize=(10, 8))
    sns.heatmap(scores_pivot, annot=True, fmt=".2f", cmap="viridis",
                cbar_kws={"label": "Cohen's d (mayor = más discriminativo)"})
    plt.title("Poder discriminativo de cada feature para cada clase")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_discrimination.png", dpi=120)
    plt.close()

    print("\nTop features más discriminativas por clase:")
    for label in label_cols:
        top = scores[scores["label"] == label].sort_values(
            "discrimination", ascending=False).head(3)
        print(f"  {label}:")
        for _, row in top.iterrows():
            print(f"    {row['feature']}: d={row['discrimination']:.2f}")


def main():
    output_dir = setup_output(OUTPUT_DIR)

    df = pd.read_csv(INPUT_CSV)
    print(f"Dataset cargado: {len(df)} filas")

    if FILTER_INVALID_POSE and "pose_valid" in df.columns:
        before = len(df)
        df = df[df["pose_valid"] == 1].reset_index(drop=True)
        print(f"Filtradas {before - len(df)} filas con pose inválida. "
              f"Análisis sobre {len(df)} frames.")

    print("\nGenerando análisis...")
    summary_statistics(df, FEATURE_COLUMNS, output_dir)
    label_distribution(df, LABEL_COLUMNS, output_dir)
    label_cooccurrence(df, LABEL_COLUMNS, output_dir)
    feature_correlation(df, FEATURE_COLUMNS, output_dir)
    histograms_by_class(df, FEATURE_COLUMNS, LABEL_COLUMNS, output_dir)
    boxplots_by_class(df, FEATURE_COLUMNS, LABEL_COLUMNS, output_dir)
    feature_discrimination_scores(df, FEATURE_COLUMNS, LABEL_COLUMNS, output_dir)

    print(f"\nAnálisis completo. Resultados en: {output_dir.absolute()}")


if __name__ == "__main__":
    main()