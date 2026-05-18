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

El enfoque es **puramente tabular**: en lugar de clasificar las imágenes
directamente con una CNN, se extrae la pose del sujeto, se calculan características
geométricas interpretables (ángulos articulares, distancias normalizadas) y se
clasifica con modelos de aprendizaje automático sobre esas características.

## Pipeline

```
Vídeos
  │
  ├─ Extracción de fotogramas + etiquetado multietiqueta (Roboflow)
  │
  ▼
Estimación de pose (YOLO11-pose) + selección de la persona principal
  │
  ▼
Cálculo de características geométricas (ángulos, distancias, interacciones fase-postura)
  │
  ▼
Clasificación multietiqueta (Random Forest / XGBoost)
  │
  ▼
Predicción de errores por fotograma
```

## Estructura del repositorio

```
.
├── scripts/
│   ├── extract_keypoints.py           # Extrae keypoints de la persona principal
│   ├── compute_features.py            # Calcula las características geométricas
│   ├── split_dataset.py               # Divide en train/val/test estratificado
│   ├── exploratory_analysis.py        # Análisis exploratorio de las features
│   ├── visual_check.py                # Verificación visual de la extracción de pose
│   ├── apply_mlsmote.py               # Oversampling de clases minoritarias (MLSMOTE)
│   ├── train_baseline.py              # Baseline con Random Forest
│   ├── train_xgboost.py               # Modelo XGBoost (modelo final)
│   └── train_xgboost_tuned.py         # XGBoost con búsqueda de hiperparámetros
├── data/                              # Datasets (no versionado, ver .gitignore)
├── output/                            # Resultados y modelos (no versionado)
├── requirements.txt
├── .gitignore
└── README.md
```

## Requisitos

- Python 3.10 o superior
- Las dependencias están en `requirements.txt`

Instalación:

```bash
pip install -r requirements.txt
```

Dependencias principales: `ultralytics` (YOLO11-pose), `xgboost`, `scikit-learn`,
`opencv-python`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `iterative-stratification`.

## Uso

El pipeline se ejecuta en orden. Antes de cada script, revisar las rutas en su
sección de configuración (`IMAGES_DIR`, rutas de entrada/salida).

### Pipeline base (genera el modelo final)

```bash
# 1. Extraer keypoints de la persona principal
python scripts/extract_keypoints.py

# 2. Calcular las características geométricas
python scripts/compute_features.py

# 3. Dividir en train / validation / test
python scripts/split_dataset.py

# 4. Entrenar el modelo final (XGBoost)
python scripts/train_xgboost.py
```

### Análisis y experimentos adicionales

```bash
# Análisis exploratorio de las características
python scripts/exploratory_analysis.py

# Verificación visual de la extracción de pose
python scripts/visual_check.py

# Baseline con Random Forest
python scripts/train_baseline.py

# Oversampling de clases minoritarias y reentrenamiento
python scripts/apply_mlsmote.py
# (luego ajustar TRAIN_CSV en train_xgboost.py al dataset aumentado)

# Búsqueda de hiperparámetros
python scripts/train_xgboost_tuned.py
```

## Características utilizadas

El modelo emplea 13 características derivadas de la pose:

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

Modelo final: **XGBoost** multietiqueta (un clasificador binario por clase, con
`scale_pos_weight` por clase y early stopping).

| Métrica | Valor (test) |
|---------|--------------|
| Macro-F1 | 0.55 |
| Micro-F1 | 0.57 |

F1 por clase (test, umbrales calibrados):

| Clase | F1 |
|-------|-----|
| Pierna | 0.65 |
| Cabeza | 0.62 |
| Dorsal | 0.57 |
| Agarre | 0.54 |
| Lumbar | 0.51 |
| Distancia | 0.40 |

## Limitaciones conocidas

- **Split no disjunto por sujeto.** Por la organización del dataset durante el
  etiquetado, no fue posible garantizar que train, validation y test contengan
  fotogramas de sujetos completamente distintos. Las métricas pueden estar
  ligeramente sobreestimadas respecto a un escenario con sujetos nuevos.
- **Techo del enfoque tabular.** Algunas clases (especialmente Distancia y
  Lumbar) tienen un rendimiento limitado. El análisis sugiere que ciertos errores
  no son completamente discriminables a partir de pose 2D.
- **Trabajo frame a frame.** El sistema clasifica fotogramas independientes y no
  modela la dimensión temporal del ejercicio.

## Trabajo futuro

- Recogida de datos con trazabilidad explícita sujeto-vídeo para un split estricto.
- Incorporación de información temporal entre fotogramas.

## Autores

Proyecto de la asignatura de Visión Artificial.
- Alejandro Sosa Corral
- Antonio Quijano Herrera
- Lucía García Lado
- Carmen Gutiérrez Silva