"""
Preprocesado del dataset: elimina las bandas negras (letterbox/pillarbox) de
todas las imágenes y guarda las versiones recortadas en una carpeta nueva.

Se hace una vez antes del pipeline para no tener que repetir la operación
en cada paso. Las imágenes originales se mantienen intactas.

También verifica visualmente unos ejemplos para confirmar que el recorte
funcionó como se espera.
"""

import shutil
from pathlib import Path
import cv2
import numpy as np

# ---------- CONFIGURACIÓN ----------
INPUT_DIR = "data_dirty"
OUTPUT_DIR = "data"

# Parámetros del recorte (mismos que veníamos usando)
BORDER_BRIGHTNESS_THRESHOLD = 15  # filas/columnas con media < esto se consideran banda
MIN_USEFUL_FRACTION = 0.3         # si tras recortar queda <30%, no recortamos

# Si True, también copia el archivo _classes.csv al directorio de salida
COPY_CLASSES_CSV = True
CLASSES_CSV_NAME = "data_dirty/_classes.csv"
# -----------------------------------


def crop_black_borders(img, threshold=BORDER_BRIGHTNESS_THRESHOLD):
    """Devuelve (imagen_recortada, hubo_recorte)."""
    if img is None or img.size == 0:
        return img, False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    col_means = gray.mean(axis=0)
    row_means = gray.mean(axis=1)

    non_black_cols = np.where(col_means > threshold)[0]
    non_black_rows = np.where(row_means > threshold)[0]

    if len(non_black_cols) == 0 or len(non_black_rows) == 0:
        return img, False

    x1 = int(non_black_cols[0])
    x2 = int(non_black_cols[-1]) + 1
    y1 = int(non_black_rows[0])
    y2 = int(non_black_rows[-1]) + 1

    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        return img, False

    useful_fraction = (cropped.shape[0] * cropped.shape[1]) / (h * w)
    if useful_fraction < MIN_USEFUL_FRACTION:
        return img, False

    was_cropped = (cropped.shape[0] != h) or (cropped.shape[1] != w)
    return cropped, was_cropped


def main():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([
        p for p in input_path.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    print(f"Procesando {len(image_paths)} imágenes...")
    print(f"  Entrada: {input_path.absolute()}")
    print(f"  Salida:  {output_path.absolute()}")

    cropped_count = 0
    failed_count = 0
    size_stats = []  # (filename, w_in, h_in, w_out, h_out)

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Aviso: no se pudo leer {img_path.name}")
            failed_count += 1
            continue

        h_in, w_in = img.shape[:2]
        cropped, was_cropped = crop_black_borders(img)
        h_out, w_out = cropped.shape[:2]

        out_file = output_path / img_path.name
        cv2.imwrite(str(out_file), cropped)

        if was_cropped:
            cropped_count += 1
            size_stats.append((img_path.name, w_in, h_in, w_out, h_out))

        if (i + 1) % 200 == 0:
            print(f"  Procesadas {i + 1}/{len(image_paths)}")

    # Copiar el _classes.csv si existe
    if COPY_CLASSES_CSV:
        classes_src = input_path / CLASSES_CSV_NAME
        if classes_src.exists():
            shutil.copy(classes_src, output_path / CLASSES_CSV_NAME)
            print(f"\nCopiado {CLASSES_CSV_NAME} al directorio de salida.")

    print(f"\n{'='*50}")
    print(f"RESUMEN")
    print(f"{'='*50}")
    print(f"Total procesadas: {len(image_paths)}")
    print(f"Recortadas:       {cropped_count} ({100*cropped_count/len(image_paths):.1f}%)")
    print(f"Sin cambios:      {len(image_paths) - cropped_count - failed_count}")
    print(f"Fallidas:         {failed_count}")

    if size_stats:
        print("\nEjemplos de recortes (primeros 5):")
        for name, w_in, h_in, w_out, h_out in size_stats[:5]:
            reduction = 100 * (1 - (w_out * h_out) / (w_in * h_in))
            print(f"  {name}")
            print(f"    {w_in}x{h_in} -> {w_out}x{h_out} (reducción {reduction:.1f}%)")


if __name__ == "__main__":
    main()