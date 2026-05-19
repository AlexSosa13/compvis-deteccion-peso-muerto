"""
Rama basada en imágenes: clasificación multietiqueta de errores en peso muerto
mediante una CNN (EfficientNet-B0 con transfer learning).

Esta rama es la contraparte end-to-end del enfoque tabular basado en pose.
Usa EXACTAMENTE los mismos splits (dataset_train/val/test.csv) para que las
métricas sean directamente comparables con la rama tabular.

Estrategia de entrenamiento en dos fases:
  Fase 1: backbone congelado, se entrena solo la cabeza de clasificación.
  Fase 2: se descongelan las últimas capas del backbone y se afina todo
          con un learning rate bajo.

Desbalance: BCEWithLogitsLoss con pos_weight por clase.
Evaluación: precision/recall/F1 por clase, macro-F1, micro-F1, igual que la
rama tabular.

Requisitos: torch, torchvision, pandas, numpy, scikit-learn, pillow, matplotlib, seaborn
"""

import os
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

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, precision_recall_curve, average_precision_score
)

# ---------- CONFIGURACIÓN ----------
IMAGES_DIR = "data"   # AJUSTAR: carpeta de imágenes preprocesadas
TRAIN_CSV = "splits/dataset_train.csv"
VAL_CSV = "splits/dataset_val.csv"
TEST_CSV = "splits/dataset_test.csv"
OUTPUT_DIR = "output/resultados_cnn"

LABEL_COLUMNS = ["Agarre", "Cabeza", "Distancia", "Dorsal", "Lumbar", "Pierna"]

# Hiperparámetros
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

# Fase 1: solo la cabeza
PHASE1_EPOCHS = 8
PHASE1_LR = 1e-3

# Fase 2: fine-tuning de las últimas capas
PHASE2_EPOCHS = 20
PHASE2_LR = 1e-4
UNFREEZE_LAST_N_BLOCKS = 3   # cuántos bloques finales del backbone descongelar

EARLY_STOPPING_PATIENCE = 6  # épocas sin mejora en val antes de parar (fase 2)

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===================== DATASET =====================

class DeadliftDataset(Dataset):
    """Dataset de fotogramas de peso muerto con etiquetas multietiqueta."""

    def __init__(self, csv_path, images_dir, label_columns, transform=None):
        self.df = pd.read_csv(csv_path)
        # Filtrar filas sintéticas si las hubiera (de MLSMOTE): la CNN solo usa imágenes reales
        if "is_synthetic" in self.df.columns:
            self.df = self.df[self.df["is_synthetic"] == 0].reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.label_columns = label_columns
        self.transform = transform

        # Verificar que las imágenes existen
        missing = []
        for fname in self.df["filename"]:
            if not (self.images_dir / fname).exists():
                missing.append(fname)
        if missing:
            print(f"  AVISO: {len(missing)} imágenes no encontradas en {images_dir}")
            print(f"    Ejemplos: {missing[:3]}")
            # Filtrar las que faltan para no romper el entrenamiento
            self.df = self.df[~self.df["filename"].isin(missing)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / row["filename"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            [row[c] for c in self.label_columns], dtype=torch.float32
        )
        return image, labels


def resize_with_padding(img_size):
    """Transform que redimensiona manteniendo el aspect ratio y rellena con negro.

    Evita la deformación que produciría un resize directo a cuadrado (las
    imágenes de peso muerto son verticales).
    """
    def _transform(img):
        w, h = img.size
        scale = img_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        # Padding hasta cuadrado
        new_img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
        new_img.paste(img, ((img_size - new_w) // 2, (img_size - new_h) // 2))
        return new_img
    return _transform


# Normalización estándar de ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Transforms de entrenamiento: con data augmentation
train_transform = transforms.Compose([
    transforms.Lambda(resize_with_padding(IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=12),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.92, 1.08)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Transforms de validación/test: sin augmentation
eval_transform = transforms.Compose([
    transforms.Lambda(resize_with_padding(IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ===================== MODELO =====================

def build_model(num_classes):
    """EfficientNet-B0 preentrenado en ImageNet con cabeza multietiqueta."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Sustituir el clasificador final por uno con num_classes salidas
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model):
    """Congela todo el backbone, deja solo la cabeza entrenable."""
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_blocks(model, n_blocks):
    """Descongela los últimos n bloques del backbone para el fine-tuning."""
    # model.features es un Sequential de bloques. Descongelamos los últimos n.
    blocks = list(model.features.children())
    for block in blocks[-n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True


# ===================== ENTRENAMIENTO =====================

def compute_pos_weights(train_csv, label_columns):
    """Calcula pos_weight por clase: ratio negativos/positivos en train."""
    df = pd.read_csv(train_csv)
    if "is_synthetic" in df.columns:
        df = df[df["is_synthetic"] == 0]
    pos_weights = []
    for c in label_columns:
        n_pos = (df[c] == 1).sum()
        n_neg = (df[c] == 0).sum()
        pos_weights.append(n_neg / max(n_pos, 1))
    return torch.tensor(pos_weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """Ejecuta una época. Si train=False, solo evalúa."""
    model.train() if train else model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return avg_loss, logits, labels


def evaluate_macro_f1(logits, labels, threshold=0.5):
    """Macro-F1 a partir de logits (aplica sigmoide internamente)."""
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    return f1_score(labels, preds, average="macro", zero_division=0)


def train_phase(model, train_loader, val_loader, criterion, optimizer,
                n_epochs, device, phase_name, early_stopping_patience=None):
    """Entrena una fase completa. Devuelve el mejor estado del modelo."""
    best_val_f1 = -1
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, n_epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion,
                                     optimizer, device, train=True)
        val_loss, val_logits, val_labels = run_epoch(model, val_loader, criterion,
                                                     optimizer, device, train=False)
        val_f1 = evaluate_macro_f1(val_logits, val_labels)

        print(f"  [{phase_name}] Época {epoch}/{n_epochs} - "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_macroF1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (early_stopping_patience is not None and
                    epochs_without_improvement >= early_stopping_patience):
                print(f"  [{phase_name}] Early stopping en época {epoch} "
                      f"(mejor val_macroF1={best_val_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_f1


# ===================== EVALUACIÓN =====================

def find_optimal_thresholds(logits, labels, label_names):
    """Umbral óptimo por clase (maximiza F1) calibrado sobre validation."""
    probs = 1 / (1 + np.exp(-logits))
    optimal = {}
    for i, name in enumerate(label_names):
        precs, recs, thresholds = precision_recall_curve(labels[:, i], probs[:, i])
        denom = precs + recs
        with np.errstate(divide="ignore", invalid="ignore"):
            f1s = np.where(denom > 0, 2 * precs * recs / denom, 0)
        best_idx = np.argmax(f1s[:-1]) if len(f1s) > 1 else 0
        optimal[name] = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
    return optimal


def evaluate_full(logits, labels, label_names, thresholds=None):
    """Métricas completas por clase. thresholds puede ser float o dict."""
    probs = 1 / (1 + np.exp(-logits))
    if thresholds is None:
        thr_arr = np.full(len(label_names), 0.5)
    elif isinstance(thresholds, dict):
        thr_arr = np.array([thresholds[c] for c in label_names])
    else:
        thr_arr = np.full(len(label_names), thresholds)

    preds = np.zeros_like(labels, dtype=int)
    for i in range(len(label_names)):
        preds[:, i] = (probs[:, i] >= thr_arr[i]).astype(int)

    rows = []
    for i, name in enumerate(label_names):
        rows.append({
            "label": name,
            "threshold": thr_arr[i],
            "support": int(labels[:, i].sum()),
            "precision": precision_score(labels[:, i], preds[:, i], zero_division=0),
            "recall": recall_score(labels[:, i], preds[:, i], zero_division=0),
            "f1": f1_score(labels[:, i], preds[:, i], zero_division=0),
            "ap": average_precision_score(labels[:, i], probs[:, i]),
        })
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    return pd.DataFrame(rows), macro_f1, micro_f1, preds


def plot_confusion_matrices(labels, preds, label_names, output_path):
    n = len(label_names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows))
    axes = axes.flatten()
    for i, name in enumerate(label_names):
        cm = confusion_matrix(labels[:, i], preds[:, i])
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
    set_seed(RANDOM_SEED)
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Dispositivo: {DEVICE}")
    if DEVICE == "cpu":
        print("  AVISO: entrenar en CPU será muy lento. Se recomienda GPU.")

    # Datasets y loaders
    print("\nCargando datasets...")
    train_ds = DeadliftDataset(TRAIN_CSV, IMAGES_DIR, LABEL_COLUMNS, train_transform)
    val_ds = DeadliftDataset(VAL_CSV, IMAGES_DIR, LABEL_COLUMNS, eval_transform)
    test_ds = DeadliftDataset(TEST_CSV, IMAGES_DIR, LABEL_COLUMNS, eval_transform)
    print(f"  Train: {len(train_ds)} imágenes")
    print(f"  Val:   {len(val_ds)} imágenes")
    print(f"  Test:  {len(test_ds)} imágenes")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # Modelo
    print("\nConstruyendo EfficientNet-B0...")
    model = build_model(len(LABEL_COLUMNS)).to(DEVICE)

    # Pérdida ponderada
    pos_weights = compute_pos_weights(TRAIN_CSV, LABEL_COLUMNS).to(DEVICE)
    print(f"pos_weights por clase: "
          f"{dict(zip(LABEL_COLUMNS, [f'{w:.2f}' for w in pos_weights.cpu().numpy()]))}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # ===== FASE 1: solo la cabeza =====
    print("\n" + "=" * 60)
    print("FASE 1: entrenamiento de la cabeza (backbone congelado)")
    print("=" * 60)
    freeze_backbone(model)
    optimizer1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE1_LR
    )
    model, f1_phase1 = train_phase(model, train_loader, val_loader, criterion,
                                   optimizer1, PHASE1_EPOCHS, DEVICE, "Fase 1")

    # ===== FASE 2: fine-tuning =====
    print("\n" + "=" * 60)
    print(f"FASE 2: fine-tuning (últimos {UNFREEZE_LAST_N_BLOCKS} bloques descongelados)")
    print("=" * 60)
    unfreeze_last_blocks(model, UNFREEZE_LAST_N_BLOCKS)
    optimizer2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE2_LR
    )
    model, f1_phase2 = train_phase(model, train_loader, val_loader, criterion,
                                   optimizer2, PHASE2_EPOCHS, DEVICE, "Fase 2",
                                   early_stopping_patience=EARLY_STOPPING_PATIENCE)

    print(f"\nMejor val macro-F1: fase 1 = {f1_phase1:.4f}, fase 2 = {f1_phase2:.4f}")

    # ===== EVALUACIÓN =====
    print("\nEvaluando en validation y test...")
    _, val_logits, val_labels = run_epoch(model, val_loader, criterion,
                                          None, DEVICE, train=False)
    _, test_logits, test_labels = run_epoch(model, test_loader, criterion,
                                            None, DEVICE, train=False)

    # Umbral 0.5
    test_metrics_05, macro_05, micro_05, _ = evaluate_full(
        test_logits, test_labels, LABEL_COLUMNS, thresholds=0.5)
    print("\n--- TEST con umbral 0.5 ---")
    print(test_metrics_05.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n  Macro-F1: {macro_05:.3f}, Micro-F1: {micro_05:.3f}")
    test_metrics_05.to_csv(out / "metrics_test_thr05.csv", index=False)

    # Umbrales óptimos (calibrados en val)
    optimal = find_optimal_thresholds(val_logits, val_labels, LABEL_COLUMNS)
    test_metrics_opt, macro_opt, micro_opt, preds_opt = evaluate_full(
        test_logits, test_labels, LABEL_COLUMNS, thresholds=optimal)
    print("\n--- TEST con umbrales óptimos por clase ---")
    print(test_metrics_opt.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n  Macro-F1: {macro_opt:.3f}, Micro-F1: {micro_opt:.3f}")
    test_metrics_opt.to_csv(out / "metrics_test_optimal_thr.csv", index=False)

    # Visualización
    plot_confusion_matrices(test_labels, preds_opt, LABEL_COLUMNS,
                            out / "confusion_matrices_test.png")

    # Guardar modelo
    torch.save({
        "model_state": model.state_dict(),
        "label_columns": LABEL_COLUMNS,
        "optimal_thresholds": optimal,
        "img_size": IMG_SIZE,
    }, out / "efficientnet_b0_deadlift.pt")
    print(f"\nModelo guardado en {out / 'efficientnet_b0_deadlift.pt'}")
    print(f"Resultados completos en: {out.absolute()}")


if __name__ == "__main__":
    main()