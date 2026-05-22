# Decisiones de diseño del proyecto

Documento de apoyo para la redacción de la memoria/presentación. Recoge las decisiones más
relevantes tomadas durante el desarrollo, su motivación y, donde procede, las
alternativas que se descartaron.

---

## 1. Formulación del problema

### Decisión: eliminar la columna "Bien" del dataset
Inicialmente las etiquetas incluían una columna "Bien" además de las seis de error.
Se eliminó para evitar inconsistencias (un fotograma con `Bien=1` y `Dorsal=1`
sería contradictorio) y por redundancia: "Bien" se infiere de las otras seis.

---

## 2. Preprocesado de las imágenes

### Decisión: detectar y eliminar bandas negras antes del pipeline
Aproximadamente la mitad de las imágenes contenían bandas negras laterales o
superiores (consecuencia de el proceso de edición de los vídeos para dividirlos en repeticiones).
Estas bandas distorsionaban el cálculo del área relativa de las personas detectadas
y reducían el rendimiento del detector. Se implementó una función que detecta
filas y columnas casi negras y las recorta, con una salvaguarda que evita recortar
más del 70% de la imagen para no fallar en frames legítimamente oscuros.

### Decisión: orientación de las imágenes resuelta externamente
Algunos vídeos estaban grabados con la cámara horizontal del ordenador y otros con la cámara del móvil
vertical. Se exploró una solución automática (probar las cuatro rotaciones y
quedarse con la que mejor pose detectaba), pero falló en torno al 16% de las
imágenes. Finalmente se rotaron todas las imágenes manualmente para garantizar
una orientación consistente antes de entrar al pipeline.

---

## 3. Estimación de pose y selección de la persona principal

### Decisión: usar YOLO11-pose (modelo top-down) en lugar de MediaPipe
YOLO11-pose detecta primero personas con sus bounding boxes y después estima la
pose dentro de cada caja. Esto era necesario porque varios vídeos contienen
personas adicionales en el fondo (incluidas deliberadamente para hacer el modelo
más robusto). Un modelo bottom-up no permitía separar fácilmente al sujeto
principal de los secundarios.

### Decisión: selección heurística de la persona principal por fotograma
Como el clasificador final trabaja frame a frame, la persona principal se identifica
también frame a frame, sin tracking temporal. La heurística combina tres criterios
normalizados:

- Tamaño relativo de la bounding box (la persona principal suele estar más cerca).
- Centralidad respecto al centro de la imagen.
- Confianza media de los keypoints detectados.

Con pesos `0.5`, `0.3` y `0.2` respectivamente. Se descartan detecciones con
menos de 8 keypoints válidos, confianza media menor a 0.45 o área menor al 3% de
la imagen. Estos parámetros se ajustaron tras inspección visual sobre una muestra
de 80 imágenes.

### Decisión: limitar el área mínima al 3% (originalmente 5%)
En ciertos fotogramas el sujeto principal aparece pequeño en la imagen (cuando
está agachado en la posición baja del peso muerto). Bajar el umbral del 5% al
3% recuperó esas detecciones sin admitir falsos positivos de personas del fondo, ya que el filtro de confianza media se subió simultáneamente a 0.45.

---

## 4. Características geométricas (rama tabular)

### Decisión: trabajar con ángulos y ratios, no coordenadas crudas
Los keypoints en píxeles dependen del tamaño y posición del sujeto en la imagen.
Las características derivadas (ángulos articulares, distancias normalizadas por
la longitud del torso) son invariantes a escala y posición. Esto permite mezclar
fotogramas de distintos sujetos a diferentes distancias de la cámara.

### Decisión: añadir características de interacción fase-postura
Durante el análisis exploratorio inicial, el ángulo de rodilla resultó casi
nulamente discriminativo para el error "Pierna" (Cohen's d = 0.01). El motivo
identificado fue que un ángulo de rodilla puede ser correcto o incorrecto según
la fase del movimiento (rodilla extendida es correcta en el bloqueo pero
incorrecta en la posición baja). Para capturar esta dependencia se añadieron tres
características que combinan ángulos con un indicador de la fase, derivado de la
altura relativa de las muñecas:

- `knee_x_phase`: producto del ángulo de rodilla por el indicador de fase.
- `neck_x_phase`: análogo para el ángulo del cuello.
- `knee_extension_low`: específica del error "pierna extendida en posición baja".

Tras añadirlas, su poder discriminativo subió notablemente (por ejemplo,
`knee_x_phase` alcanzó Cohen's d = 0.62 para Pierna y 0.82 para Lumbar).

### Decisión: eliminar `torso_horizontal` por redundancia
Esta característica tenía correlación -1.0 con `back_angle_vertical` (eran la
misma información expresada de dos formas). Se eliminó para evitar redundancia
y reducir la dimensionalidad sin perder señal.

---

## 5. División del dataset

### Decisión: estratificación multietiqueta
Para problemas multietiqueta, la estratificación clásica de scikit-learn no
funciona. Se usó la biblioteca `iterative-stratification`, que mantiene la
distribución de cada clase aproximadamente constante entre los splits, incluyendo
las minoritarias. Esto es importante porque "Distancia" representa solo el 5%
del dataset.

### Decisión: split aleatorio a nivel de fotograma, no por vídeo
Esta es una limitación del proyecto. El ideal sería no juntar el mismo vídeo en varios splits (todos
los fotogramas de un vídeo en un mismo split), pero el dataset llegó organizado
por repeticiones individuales con nombres genéricos en algunos casos (`1.mp4`, `2.mp4`, etc.),
sin trazabilidad clara al vídeo original. Recuperar esa información habría
requerido reorganizar el dataset entero. Se asumió la limitación y se documentó.

### Decisión: ratio 70/15/15 con semilla fija (42)
Estándar y suficiente para el tamaño del dataset. La semilla se fija para
reproducibilidad. El test queda con 26 ejemplos de Distancia y 66 de Agarre, lo
que implica una varianza notable en las métricas de las clases minoritarias.

---

## 6. Manejo del desbalance de clases

### Decisión: class weights, no oversampling generalizado
Se evaluaron ambas opciones. Class weights (`class_weight='balanced'` en Random
Forest, `scale_pos_weight` en XGBoost, `pos_weight` en BCE para la CNN) se
priorizaron por ser más simples, no introducir muestras sintéticas y aplicarse
durante el entrenamiento sin alterar los datos.

### Decisión: MLSMOTE como experimento separado
Se probó MLSMOTE en dos variantes (agresiva y conservadora) sobre el split de
train para aumentar las clases minoritarias. La variante agresiva (K=5 vecinos,
400 muestras objetivo, propagación de etiquetas por votación) **empeoró** los
resultados al arrastrar etiquetas espurias a clases correlacionadas. La variante
conservadora (K=3 vecinos, 200 muestras, etiqueta solo de la clase minoritaria
objetivo) **igualó** los resultados sin oversampling. La conclusión fue que el
problema de Distancia no es por falta de muestras, sino por limitación intrínseca
de las características para discriminar esa clase.

### Decisión: calibración de umbrales por clase sobre validation
El umbral 0.5 por defecto no es óptimo cuando las clases están desbalanceadas
y se usan class weights (el modelo tiende a sobre-predecir positivos). Para
cada clase, se busca el umbral que maximiza F1 sobre validation y se aplica a
test. Esto mejora consistentemente el macro-F1 entre 5 y 9 puntos respecto al
umbral 0.5.

---

## 7. Rama tabular

### Decisión: XGBoost como modelo final, sobre Random Forest
Random Forest se usó como baseline inicial (macro-F1 = 0.53). XGBoost mejoró
hasta 0.55 con los mismos datos. El análisis de la importancia de características
reveló que el Random Forest repartía la importancia uniformemente entre todas
las características, mientras que XGBoost se concentraba en las más
discriminativas. Esto sugiere que XGBoost explota mejor las características de
interacción fase-postura.

### Decisión: un modelo binario por clase, no MultiOutputClassifier directo
Cada clase tiene sus propios `scale_pos_weight` y mejor número de iteraciones
con early stopping. `MultiOutputClassifier` no permite estas configuraciones
específicas por clase, así que se implementó manualmente un modelo por clase.

### Decisión: descartar la búsqueda de hiperparámetros
Se probó un random search de 40 combinaciones por clase (240 modelos en total).
El resultado fue **peor** que el XGBoost con hiperparámetros por defecto en test
(0.542 vs 0.548). El motivo identificado fue overfitting al validation set:
con solo 26 ejemplos de Distancia en val, el grid search seleccionó
configuraciones que generalizaban mal. Se mantuvo el XGBoost sin tunear como
modelo final.

---

## 8. Rama de imagen (CNN)

### Decisión: transfer learning con EfficientNet-B0
Con ~2500 imágenes de entrenamiento, entrenar una CNN desde cero no era viable.
Se eligió EfficientNet-B0 por su buena relación rendimiento/tamaño y su rapidez
de entrenamiento. Se descartaron alternativas más pesadas (ResNet50, ConvNeXt)
para evitar overfitting con el tamaño de dataset disponible.

### Decisión: fine-tuning en dos fases
- **Fase 1** (8 épocas): backbone congelado, solo se entrena la cabeza de
  clasificación. Permite que la cabeza encuentre un buen punto de partida sin
  perturbar los pesos preentrenados.
- **Fase 2** (hasta 20 épocas con early stopping): se descongelan los 3 últimos
  bloques del backbone y se afinan con learning rate bajo (1e-4). Permite que
  el modelo adapte sus capas más específicas al dominio del peso muerto.

### Decisión: resize con padding, no resize directo
Las imágenes son verticales (1920×1080 girado). Un resize directo a 224×224
las deformaría y modificaría los ángulos del cuerpo, que es justo lo que el
modelo debe juzgar. Se usa resize manteniendo la relación de aspecto y
rellenando con negro hasta el cuadrado.

### Decisión: data augmentation moderada
Flip horizontal (el peso muerto es simétrico), rotación ±12°, jitter de color
y pequeñas traslaciones y zoom. Se descartaron rotaciones grandes porque
distorsionarían la geometría que el modelo debe aprender.

### Decisión: BCEWithLogitsLoss con pos_weight por clase
Equivalente al `scale_pos_weight` de XGBoost, asegura que las clases minoritarias
(especialmente Distancia, con pos_weight ≈ 20) reciben suficiente atención durante
el entrenamiento.

---

## 9. Rama de fusión

### Decisión: combinar rama tabular y rama CNN
Las dos ramas cometen errores distintos: la tabular es competitiva en Cabeza
(donde los ángulos calculados ya capturan bien la información), mientras que
la CNN es claramente superior en Agarre y Distancia (donde la señal está en
píxeles finos que la pose 2D no captura). La fusión las combina.

### Decisión: meta-modelo calibrado sobre validation, no sobre train
Las dos ramas base se entrenaron con train. Si el meta-modelo se calibrase sobre
train, las predicciones de las ramas estarían infladas en ese conjunto (las
ramas ya lo habían visto) y el meta-modelo aprendería con datos engañosos. La
calibración sobre validation, donde las ramas no se entrenaron, evita esa fuga.
La evaluación final es sobre test.

### Decisión: tres estrategias de fusión probadas
- **Promedio simple** de probabilidades: macro-F1 = 0.648. Baseline sin parámetros.
- **Promedio ponderado por clase** con peso óptimo en val: macro-F1 = 0.650.
- **Stacking** con regresión logística por clase: macro-F1 = 0.651.

Las tres dan prácticamente lo mismo (dentro del margen de ruido). Se recomienda
el **promedio ponderado** como configuración final por su mejor relación entre
rendimiento e interpretabilidad: los pesos por clase son explicables, mientras
que el stacking entrena 6 regresiones logísticas sobre validation pequeña con
mayor riesgo de overfitting.

### Decisión: usar el XGBoost sin tunear (no el tuneado) en la fusión
El XGBoost tuneado había usado validation intensivamente durante el grid search,
de modo que sus predicciones sobre val están infladas. Usarlo en la fusión
contaminaría la calibración del meta-modelo. El XGBoost sin tunear usa val solo
para early stopping (uso leve) y por tanto mantiene la integridad de la
calibración.

---

## 10. Métricas y evaluación

### Decisión: macro-F1 como métrica principal
Con clases desbalanceadas, la accuracy es engañosa. F1 por clase muestra el
rendimiento individual. Macro-F1 (media no ponderada) refleja el rendimiento
general sin que las clases mayoritarias dominen. Micro-F1 se reporta también
como complemento.

### Decisión: evaluar siempre en el mismo test
Todos los modelos se evalúan sobre el mismo `dataset_test.csv` con la misma
semilla. Esto garantiza que las comparaciones entre modelos son justas y
directamente interpretables.

### Decisión: reportar matriz de confusión por clase
Cada clase binaria tiene su matriz de confusión 2×2. Esto permite ver el
patrón de errores de cada modelo: si tiende a falsos positivos o falsos
negativos, y en qué clases.

---

## 11. Resumen de resultados

Macro-F1 en test, con umbrales calibrados por clase sobre validation:

| Modelo | Macro-F1 |
|--------|----------|
| Random Forest (tabular, baseline) | 0.530 |
| XGBoost (tabular) | 0.548 |
| XGBoost + MLSMOTE conservador | 0.543 |
| XGBoost tuneado | 0.542 |
| EfficientNet-B0 (CNN) | 0.638 |
| Fusión tabular + CNN (promedio ponderado) | 0.650 |
| **Fusión tabular + CNN (stacking)** | **0.651** |

Conclusión principal: la combinación de información geométrica e información
visual mejora sobre cualquiera de las dos ramas por separado. La CNN aporta
sobre todo en errores con señal en detalles visuales finos (Agarre, Distancia);
la rama tabular contribuye con la información geométrica explícita y, sobre todo,
interpretable.

---

## 12. Limitaciones identificadas

- Splits no disjuntos por vídeo.
- La clase Distancia tiene un techo bajo en todas las aproximaciones.
- Trabajo frame a frame, sin modelado temporal del ejercicio.
- Validation pequeño limita la complejidad del meta-modelo de fusión.

---

## 13. Trabajo futuro

- Recogida de datos con trazabilidad sujeto-vídeo explícita.
- Modelos secuenciales que integren información temporal.
- Ampliación del dataset, especialmente para Distancia.
- Backbones de CNN más recientes o más pesados conforme aumente el dataset.
