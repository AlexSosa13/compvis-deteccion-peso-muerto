"""
Interfaz de demostración del modelo final.

Acepta una imagen o un vídeo y muestra la pose detectada junto con las
predicciones del modelo de fusión (XGBoost + EfficientNet-B0) sobre cada frame.

Ejecución:
    python demo_app.py
    python demo_app.py --share   # enlace público temporal de Gradio
"""

import argparse

import cv2
import numpy as np
import pandas as pd
import gradio as gr
from ultralytics import YOLO

from src.constants import LABEL_COLUMNS
from src.preprocessing import crop_black_borders
from src.pose import select_main_person, draw_pose_on_image
from src.features import compute_features
from src.models import (
    load_xgboost, load_cnn, load_fusion_config,
    predict_tabular, predict_cnn, fuse_probabilities,
)

# ---------- CONFIGURACIÓN ----------
XGBOOST_MODEL = "output/resultados_xgboost/xgboost_models.joblib"
CNN_MODEL = "output/resultados_cnn/efficientnet_b0_deadlift.pt"
FUSION_CONFIG = "output/resultados_fusion/fusion_config.joblib"
YOLO_MODEL = "yolo11n-pose.pt"
# -----------------------------------


print("Cargando modelos...")
yolo_model = YOLO(YOLO_MODEL)
xgb_models = load_xgboost(XGBOOST_MODEL)
cnn_model = load_cnn(CNN_MODEL)
FUSION_WEIGHTS, FUSION_THRESHOLDS = load_fusion_config(FUSION_CONFIG)
print("Modelos cargados.\n")


def build_predictions_table(p_tab, p_cnn, p_fused):
    """Construye la tabla de predicciones con detección final."""
    rows = []
    for c in LABEL_COLUMNS:
        thr = FUSION_THRESHOLDS[c]
        detectado = "✅ Sí" if p_fused[c] >= thr else "—"
        rows.append({
            "Clase": c,
            "P(tabular)": f"{p_tab[c]:.3f}",
            "P(CNN)": f"{p_cnn[c]:.3f}",
            "P(fusión)": f"{p_fused[c]:.3f}",
            "Umbral": f"{thr:.3f}",
            "Detectado": detectado,
        })
    return pd.DataFrame(rows)


def process_frame(img_bgr):
    """Procesa un frame BGR. Devuelve (img anotada RGB, tabla, mensaje)."""
    if img_bgr is None or img_bgr.size == 0:
        return None, pd.DataFrame(), "No se pudo leer el frame."

    img = crop_black_borders(img_bgr)
    h, w = img.shape[:2]
    result = yolo_model(img, verbose=False)[0]
    selection = select_main_person(result, w, h)

    if selection is None:
        msg = "No se ha detectado una persona válida en este frame."
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), pd.DataFrame(), msg

    kp_xy, kp_conf, bbox = selection
    annotated_bgr = draw_pose_on_image(img, kp_xy, kp_conf, bbox)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    features = compute_features(kp_xy, kp_conf)
    if features is None:
        return annotated_rgb, pd.DataFrame(), "Pose degenerada, no se pudieron calcular features."

    p_tab = predict_tabular(xgb_models, features)
    p_cnn = predict_cnn(cnn_model, img)
    p_fused = fuse_probabilities(p_tab, p_cnn, FUSION_WEIGHTS)

    df = build_predictions_table(p_tab, p_cnn, p_fused)
    detected = [c for c in LABEL_COLUMNS if p_fused[c] >= FUSION_THRESHOLDS[c]]
    if not detected:
        msg = "Técnica correcta: no se han detectado errores."
    else:
        msg = f"Errores detectados ({len(detected)}): {', '.join(detected)}"
    return annotated_rgb, df, msg


def extract_frames(video_path, fps_target=1.0):
    """Extrae frames de un vídeo a la tasa indicada. Devuelve lista de BGR."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, int(round(fps_video / fps_target)))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def process_image_input(img_input):
    """Handler para la pestaña de imagen."""
    if img_input is None:
        return None, pd.DataFrame(), "Sube una imagen."
    img_bgr = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)
    return process_frame(img_bgr)


def process_video_input(video_path, fps_target):
    """Handler para la pestaña de vídeo."""
    if video_path is None:
        return [], pd.DataFrame(), "Sube un vídeo."

    frames = extract_frames(video_path, fps_target=fps_target)
    if not frames:
        return [], pd.DataFrame(), "No se pudieron extraer frames del vídeo."

    gallery = []
    all_rows = []
    for i, frame in enumerate(frames):
        annotated, df, _ = process_frame(frame)
        if annotated is not None:
            annotated = annotated.copy()
            cv2.putText(annotated, f"Frame {i+1}/{len(frames)}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            gallery.append(annotated)
        if not df.empty:
            df_copy = df.copy()
            df_copy.insert(0, "Frame", i + 1)
            all_rows.append(df_copy)

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    return gallery, combined, f"Procesados {len(gallery)} de {len(frames)} frames."


def build_interface():
    with gr.Blocks(title="Detección de errores en peso muerto") as demo:
        gr.Markdown(
            "# Detección de errores posturales en peso muerto\n"
            "Demostración del modelo de fusión (XGBoost sobre pose + EfficientNet-B0). "
            "Acepta imágenes o vídeos. Para vídeos se extrae 1 frame por segundo por defecto."
        )

        with gr.Tabs():
            with gr.Tab("Imagen"):
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(label="Imagen de entrada", type="numpy")
                        img_btn = gr.Button("Analizar", variant="primary")
                    with gr.Column():
                        img_output = gr.Image(label="Pose detectada")
                        img_msg = gr.Textbox(label="Resumen", interactive=False)
                        img_table = gr.Dataframe(
                            label="Predicciones por clase",
                            headers=["Clase", "P(tabular)", "P(CNN)", "P(fusión)",
                                     "Umbral", "Detectado"],
                            wrap=True,
                        )
                img_btn.click(process_image_input,
                              inputs=[img_input],
                              outputs=[img_output, img_table, img_msg])

            with gr.Tab("Vídeo"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Vídeo de entrada")
                        fps_slider = gr.Slider(
                            minimum=0.5, maximum=4.0, value=1.0, step=0.5,
                            label="Frames por segundo a extraer",
                        )
                        video_btn = gr.Button("Analizar vídeo", variant="primary")
                    with gr.Column():
                        video_msg = gr.Textbox(label="Resumen", interactive=False)
                        video_gallery = gr.Gallery(
                            label="Frames analizados", columns=2, height="auto",
                        )
                        video_table = gr.Dataframe(
                            label="Predicciones por frame", wrap=True,
                        )
                video_btn.click(process_video_input,
                                inputs=[video_input, fps_slider],
                                outputs=[video_gallery, video_table, video_msg])

        gr.Markdown(
            "**Modelo**: fusión por promedio ponderado de las ramas tabular y CNN. "
            "Los umbrales por clase se calibraron sobre el conjunto de validación."
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true",
                        help="Expone un enlace público temporal de Gradio.")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_interface()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
