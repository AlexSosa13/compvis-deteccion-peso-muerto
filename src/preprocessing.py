"""Preprocesado de imágenes: eliminación de bandas negras."""

import cv2
import numpy as np

from .constants import BORDER_BRIGHTNESS_THRESHOLD, MIN_USEFUL_FRACTION


def crop_black_borders(img, threshold=BORDER_BRIGHTNESS_THRESHOLD):
    """Detecta y elimina bandas negras de los bordes de una imagen.

    Args:
        img: imagen BGR (numpy array).
        threshold: brillo medio máximo para considerar una fila/columna como negra.

    Returns:
        Imagen recortada (BGR). Si el recorte resultante sería demasiado pequeño
        (< MIN_USEFUL_FRACTION del área original) se devuelve la imagen original.
    """
    if img is None or img.size == 0:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    col_means = gray.mean(axis=0)
    row_means = gray.mean(axis=1)

    non_black_cols = np.where(col_means > threshold)[0]
    non_black_rows = np.where(row_means > threshold)[0]

    if len(non_black_cols) == 0 or len(non_black_rows) == 0:
        return img

    x1, x2 = int(non_black_cols[0]), int(non_black_cols[-1]) + 1
    y1, y2 = int(non_black_rows[0]), int(non_black_rows[-1]) + 1

    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        return img

    useful = (cropped.shape[0] * cropped.shape[1]) / (h * w)
    if useful < MIN_USEFUL_FRACTION:
        return img

    return cropped
