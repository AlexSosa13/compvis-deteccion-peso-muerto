# Detección de errores posturales en peso muerto mediante visión artificial

Sistema de clasificación multietiqueta que detecta errores de técnica durante
la ejecución del peso muerto a partir de fotogramas de vídeo. Proyecto desarrollado
para la asignatura de Visión Artificial.

## Descripción

El sistema analiza fotogramas de una persona realizando peso muerto e identifica
hasta seis tipos de error de forma simultánea:

- **Agarre** — posición o anchura de manos incorrecta
- **Cabeza** — posición de la cabeza/cuello incorrecta
- **Distancia** — barra demasiado alejada del cuerpo
- **Dorsal** — redondeo de la zona dorsal de la espalda
- **Lumbar** — redondeo de la zona lumbar de la espalda
- **Pierna** — flexión de rodilla incorrecta

Si no se detecta ninguno de estos errores, el fotograma se considera técnica
correcta ("Bien").

El proyecto explora y compara **tres aproximaciones** al problema:

1. **Rama tabular** — extrae la pose del sujeto, calcula características
   geométricas interpretables (ángulos, distancias) y clasifica con modelos
   de aprendizaje automático sobre esas características.
2. **Rama de imagen (CNN)** — clasifica los fotogramas de extremo a extremo
   con una red convolucional (EfficientNet-B0) mediante transfer learning.
3. **Rama de fusión** — combina las predicciones de las dos ramas anteriores,
   aprovechando la información complementaria de cada una.

Adicionalmente, el repositorio incluye una **aplicación de demostración**
interactiva que permite probar el modelo final sobre imágenes o vídeos.

## Pipeline

```
Vídeos
  │
  ├─ Extracción de fotogramas + etiquetado multietiqueta (Roboflow)
  │
  ▼
Preprocesado: eliminación de bandas negras
  │
  ├──────────────────────────────┬─────────────────────────────┐
  ▼                              ▼                             │
RAMA TABULAR                  RAMA IMAGEN                       │
  │                              │                             │
Estimación de pose            EfficientNet-B0                   │
(YOLO11-pose) +               (transfer learning,               │
selección persona             fine-tuning en 2 fases)           │
principal                        │                             │
  │                              │                             │
Características                   │                             │
geométricas                      │                             │
  │                              │                             │
Clasificador                  Clasificador                      │
(XGBoost)                     (CNN)                              │
  │                              │                             │
  └──────────────┬───────────────┘                             │
                 ▼                                              │
          RAMA DE FUSIÓN  ◄─────────────────────────────────────┘
          (combina probabilidades de ambas ramas)
                 │
                 ▼
       Predicción final de errores por fotograma
```

## Estructura del repositorio

```
.
├── src/                                # Paquete con utilidades compartidas
│   ├── __init__.py
│   ├── constants.py                    # Constantes (clases, features, parámetros) (refactor)
│   ├── preprocessing.py                # Eliminación de bandas negras (refactor)
│   ├── pose.py                         # Selección de persona principal y dibujo (refactor)
│   ├── features.py                     # Cálculo de las 13 features geométricas (refactor)
│   └── models.py                       # Carga de modelos y predicción (refactor)
│
├── scripts_old/                        # Scripts del pipeline de entrenamiento original
│   ├── preprocess_remove_borders.py    # Elimina bandas negras del dataset
│   ├── extract_keypoints.py            # Extrae keypoints de la persona principal
│   ├── compute_features.py             # Calcula las características geométricas
│   ├── split_dataset.py                # Divide en train/val/test estratificado
│   ├── exploratory_analysis.py         # Análisis exploratorio de las features
│   ├── visual_check.py                 # Verificación visual de la extracción de pose
│   ├── apply_mlsmote.py                # Oversampling de clases minoritarias (experimento)
│   ├── train_baseline.py               # Rama tabular: baseline Random Forest
│   ├── train_xgboost.py                # Rama tabular: modelo XGBoost (el final)
│   ├── train_xgboost_tuned.py          # Rama tabular: XGBoost con grid search (experimento)
│   ├── train_cnn.py                    # Rama imagen: CNN EfficientNet-B0
│   ├── train_fusion.py                 # Rama fusión: combina tabular + CNN
│
├── demo_app.py                         # Aplicación de demostración (Gradio)
├── data/                               # Datasets (no versionado, ver .gitignore)
├── output/                             # Resultados y modelos (no versionado)
├── requirements.txt
├── .gitignore
└── README.md
```

### Sobre el paquete `src/`

Los módulos en `src/` contienen el código refactorizado común utilizado tanto por
los scripts de entrenamiento como por la aplicación de demostración. Centralizar el
preprocesado, la selección de la persona principal, el cálculo de features y la
carga de modelos en un único lugar evita inconsistencias entre entrenamiento e
inferencia.

No obstante, por reproducibilidad y trasparencia, se han mantenido los códigos originales
ejecutados para obtener los resultados estadísticos reportados en la carpeta `scripts_old/`.

Los scripts deben ejecutarse desde la raíz del proyecto, no desde dentro de
`scripts_old/` o `src/`, para que los imports relativos funcionen correctamente.

## Requisitos

- Python 3.10 o superior
- GPU recomendada para la rama CNN y para la app de demostración (la rama
  tabular y los scripts de análisis funcionan en CPU)
- Las dependencias están en `requirements.txt`

Instalación:

```bash
pip install -r requirements.txt
```

Para la rama CNN se necesita PyTorch con soporte CUDA. Consultar el comando
de instalación específico para la versión de CUDA del sistema en la web oficial
de PyTorch.

## Uso

Todos los comandos se ejecutan desde la raíz del proyecto. Antes de cada
script, revisar las rutas en su sección de configuración (`IMAGES_DIR`, rutas
de entrada/salida).

### 1. Preparación de datos (común a todas las ramas)

```bash
# Eliminar bandas negras de las imágenes
python scripts_old/preprocess_remove_borders.py

# Extraer keypoints de la persona principal
python scripts_old/extract_keypoints.py

# Calcular las características geométricas
python scripts_old/compute_features.py

# Dividir en train / validation / test (estratificado multietiqueta)
python scripts_old/split_dataset.py
```

### 2. Rama tabular

```bash
# Baseline con Random Forest
python scripts_old/train_baseline.py

# Modelo XGBoost (el utilizado en la fusión)
python scripts_old/train_xgboost.py
```

### 3. Rama de imagen

```bash
# CNN EfficientNet-B0 con transfer learning
python scripts_old/train_cnn.py
```

### 4. Rama de fusión

```bash
# Combina las predicciones de la rama tabular y la CNN
python scripts_old/train_fusion.py
```

### 5. Aplicación de demostración

Una vez entrenados los tres modelos anteriores, se puede lanzar la app:

```bash
# Servidor local en http://localhost:7860
python demo_app.py

# Con enlace público temporal (útil al presentar desde otro equipo)
python demo_app.py --share
```

La aplicación tiene dos pestañas:

- **Imagen**: subir una imagen y obtener al instante la pose detectada y las
  predicciones de cada clase (probabilidad por rama, probabilidad fusionada,
  umbral y decisión final).
- **Vídeo**: subir un vídeo, elegir cuántos frames por segundo extraer
  (entre 0.5 y 4), y obtener una galería con cada frame anotado más una tabla
  con las predicciones por frame.

La aplicación reutiliza los módulos de `src/`, garantizando que la inferencia
en demostración usa exactamente el mismo preprocesado y la misma lógica de
features que durante el entrenamiento y la evaluación.

### Análisis y experimentos adicionales

```bash
# Análisis exploratorio de las características
python scripts_old/exploratory_analysis.py

# Verificación visual de la extracción de pose
python scripts_old/visual_check.py

# Oversampling de clases minoritarias (experimento)
python scripts_old/apply_mlsmote.py

# Búsqueda de hiperparámetros de XGBoost (experimento)
python scripts_old/train_xgboost_tuned.py
```

## Características utilizadas (rama tabular)

El modelo tabular emplea 13 características derivadas de la pose:

**Ángulos y distancias básicas:** ángulo de la espalda respecto a la vertical,
ángulo del cuello, ángulo de la rodilla, ángulo de la cadera, anchura de agarre
relativa a hombros y a caderas, distancia barra-tibia, altura relativa de las
muñecas, confianza media de la pose y lateralidad detectada.

**Interacciones fase-postura:** dado que un mismo ángulo puede ser correcto o
incorrecto según el momento del movimiento (por ejemplo, una pierna extendida es
correcta en el bloqueo pero incorrecta en la posición baja), se añaden tres
características que combinan ángulos con un indicador de la fase del ejercicio:
`knee_x_phase`, `neck_x_phase` y `knee_extension_low`.

## Resultados

Comparativa de las tres aproximaciones (macro-F1 en el conjunto de test,
con umbrales calibrados por clase sobre validation):

| Modelo | Macro-F1 | Micro-F1 |
|--------|----------|----------|
| Random Forest (tabular, baseline) | 0.53 | — |
| XGBoost (tabular) | 0.55 | 0.58 |
| EfficientNet-B0 (CNN) | 0.64 | 0.64 |
| **Fusión tabular + CNN** | **0.65** | **0.67** |

F1 por clase del modelo de fusión (test, umbrales calibrados):

| Clase | F1 |
|-------|-----|
| Pierna | 0.82 |
| Agarre | 0.69 |
| Cabeza | 0.65 |
| Dorsal | 0.63 |
| Lumbar | 0.59 |
| Distancia | 0.53 |

La rama de imagen supera a la tabular, especialmente en errores cuya señal
discriminante reside en detalles visuales finos (Agarre, Distancia). La rama
de fusión mejora sobre ambas, confirmando que la información geométrica y la
visual son complementarias.

## Limitaciones conocidas

- **Split no disjunto por vídeo.** Por la organización del dataset durante el
  etiquetado, no fue posible garantizar que train, validation y test contengan
  fotogramas de vídeos completamente distintos.
- **Clase Distancia.** Es la clase con menos ejemplos y la de peor rendimiento
  en todas las aproximaciones. Su detección fiable requeriría más datos.
- **Trabajo frame a frame.** El sistema clasifica fotogramas independientes y no
  modela la dimensión temporal del ejercicio.
- **Validation reducido para la fusión.** El meta-modelo de fusión se calibra
  sobre un conjunto de validación pequeño, lo que limita la complejidad de las
  estrategias de combinación que pueden emplearse sin sobreajuste.

## Trabajo futuro

- Recogida de datos con trazabilidad explícita sujeto-vídeo para un split estricto.
- Incorporación de información temporal entre fotogramas (modelos secuenciales).
- Ampliación del dataset, en particular de las clases minoritarias.
- Exploración de backbones más recientes para la rama de imagen.

## Autores

Proyecto de la asignatura de Visión Artificial.
- Alejandro Sosa Corral
- Lucía García Lado
- Antonio Quijano Herrera
- Carmen Gutiérrez Silva