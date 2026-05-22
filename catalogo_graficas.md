# Catálogo de gráficas para la memoria/presentación

Documento de apoyo para la redacción. Describe cada gráfica disponible, qué
muestra, y qué se puede decir sobre
ella.

---

## Análisis exploratorio del dataset

### Distribución de etiquetas y errores simultáneos por fotograma
**Archivo:** `analisis_exploratorio/label_distribution.png`

**Qué muestra:** dos paneles. El primero, un gráfico de barras con el número
de fotogramas que contienen cada etiqueta y los que se consideran técnica
correcta ("Bien" implícito). El segundo, la distribución de cuántos errores
simultáneos coexisten en un mismo fotograma.

**Qué destacar:**
- El dataset tiene desbalance evidente: la clase "Distancia" representa solo el
  ~5% del total, mientras que "Dorsal" y "Cabeza" superan el 24%.
- El 29.8% de los fotogramas contienen 1 error simultáneo, el 21.3% contienen 2,
  y casi un 10% contienen 3 o más. Esto justifica formalmente que el problema
  es multietiqueta, no multiclase.
- Hay un 38.5% de fotogramas sin ningún error (técnica correcta).

---

### Co-ocurrencia de etiquetas
**Archivo:** `analisis_exploratorio/label_cooccurrence.png`

**Qué muestra:** matriz simétrica donde cada celda indica cuántas veces dos
etiquetas aparecen juntas en el mismo fotograma. La diagonal muestra el total
por clase.

**Qué destacar:**
- Algunos pares tienen co-ocurrencia muy alta: Dorsal+Pierna (310 frames),
  Cabeza+Lumbar (301 frames). Tiene sentido físico: errores técnicos suelen
  manifestarse en cadena.
- Distancia y Lumbar coinciden solo en 40 frames, a pesar de que un error de
  distancia provoca habitualmente compensación lumbar. Esto sugiere
  inconsistencia entre etiquetadores: probablemente algunos marcaron la causa
  (Distancia) y otros la compensación (Lumbar). Es una limitación del dataset
  que conviene mencionar.

---

### Poder discriminativo de cada feature por clase
**Archivo:** `analisis_exploratorio/feature_discrimination.png`

**Qué muestra:** mapa de calor con el efecto de Cohen (d) de cada feature
respecto a cada clase. Valores altos (> 0.8) indican que esa feature separa
bien los frames con esa etiqueta activa de los que no la tienen.

**Qué destacar:**
- Para Lumbar, varias features de fase superan d = 0.80 (`knee_x_phase` = 0.82,
  `wrist_height_ratio` = 0.81, `neck_x_phase` = 0.80, `knee_extension_low` = 0.80).
  Las features con interacción fase-postura dominan claramente.
- Para Pierna, `knee_x_phase` (0.62) y `knee_extension_low` (0.61) son las
  mejores, mientras que `knee_angle` sin contextualizar tenía d ≈ 0.01: prueba
  visual de que considerar la fase del movimiento era necesario.
- Agarre es la clase con peor discriminación general: ninguna feature supera
  d = 0.37. Esto anticipa que la rama tabular tendrá dificultad con esta clase,
  algo que se confirma en los resultados.

---

### Correlación entre features
**Archivo:** `analisis_exploratorio/feature_correlation.png`

**Qué muestra:** matriz de correlación de Pearson entre todas las features.

**Qué destacar:**
- `back_angle_vertical` y `torso_horizontal` tenían correlación -1.0 (eran la
  misma información expresada de dos formas). Justifica visualmente la
  decisión de eliminar `torso_horizontal`.
- Las features de fase (`knee_x_phase`, `neck_x_phase`, `knee_extension_low`)
  están correlacionadas entre sí, lo que era esperable porque comparten un
  factor común (el indicador de fase). Esta correlación moderada es aceptable
  porque cada una mide un aspecto distinto.

---

## Preprocesado y estimación de pose

### Verificación visual de la pose
**Archivos:** ejemplos en `verificacion_visual/check_*.jpg`

**Qué muestra:** imágenes anotadas con la bounding box de la persona
seleccionada (verde), las bounding boxes descartadas (rojo) con su motivo, el
esqueleto de la pose dibujado en la persona principal y un marcador del centro
de la imagen.

**Qué destacar:**
- La heurística de selección de la persona principal funcionó correctamente en
  la práctica totalidad de los frames después de calibrar los parámetros.
- Es buena idea elegir 2 o 3 imágenes representativas:
  - Una con el sujeto principal claramente identificado y otra persona
    descartada por baja área.
  - Una con el sujeto en la posición baja del peso muerto, para mostrar que la
    heurística funciona incluso cuando el área del sujeto es pequeña.

---

## Rama tabular

### Importancia de features según el modelo XGBoost
**Archivo:** `rama_tabular/feature_importance.png`

**Qué muestra:** mapa de calor con la importancia (normalized gain) de cada
feature para cada clasificador binario, una fila por clase.

**Qué destacar:**
- A diferencia del Random Forest (que repartía importancia uniformemente),
  XGBoost concentra el peso en pocas features por clase. Esto coincide con lo
  que predecía el análisis de Cohen's d.
- Para Pierna, `knee_extension_low` tiene importancia 0.18 (la más alta del
  modelo), lo que confirma que la feature de interacción específicamente
  diseñada para "pierna recta en posición baja" es la más relevante.
- Para Lumbar, las tres features de fase (`knee_x_phase`, `neck_x_phase`,
  `knee_extension_low`) son las top tres del modelo, validando empíricamente
  el diseño de features.

---

### Matrices de confusión del modelo XGBoost
**Archivo:** `rama_tabular/confusion_matrices_xgboost.png`

**Qué muestra:** una matriz de confusión 2×2 por clase, sobre el conjunto de
test, con los umbrales óptimos por clase aplicados.

**Qué destacar:**
- Pierna y Cabeza son las clases con mejor balance precision/recall.
- Distancia tiene pocos verdaderos positivos (10 de 26) debido al pequeño
  número de muestras y a la dificultad intrínseca de la clase.
- En clases como Lumbar y Dorsal, el modelo tiene tendencia a sobre-predecir
  positivos (muchos falsos positivos), efecto del `scale_pos_weight` elevado.

---

### Curvas precision-recall del modelo XGBoost
**Archivo:** `rama_tabular/precision_recall_curves_xgboost.png`

**Qué muestra:** una curva precision-recall por clase, sobre test, con el AP
(average precision) en la leyenda.

**Qué destacar:**
- Pierna tiene la curva más alta (AP ≈ 0.73), indicando buen rendimiento en
  todo el rango de umbrales.
- Distancia tiene una curva con AP bajo (≈ 0.29), reflejando la dificultad de
  la clase incluso con los mejores ajustes posibles.
- La forma de las curvas justifica el uso de umbrales calibrados por clase en
  lugar de un umbral fijo en 0.5.

---

### Convergencia de XGBoost por clase
**Archivo:** `rama_tabular/xgboost_convergencia.png`

**Qué muestra:** una rejilla 2×3 con seis paneles, uno por clase. Cada panel
muestra el log-loss en validation a lo largo de las rondas de boosting, con
una línea vertical marcando el punto en que actuó el early stopping.

**Qué destacar:**
- Cada clase converge a un ritmo distinto. Cabeza paró en torno a la ronda 220,
  Distancia en 310, Agarre llegó casi al límite de las 500 rondas.
- El comportamiento de Agarre (que casi no se detuvo) sugiere que podría haber
  seguido aprendiendo con más rondas. Es una observación honesta a documentar.
- La convergencia es estable en todas las clases: no se observan oscilaciones
  importantes, lo que indica un entrenamiento sano.

---

## Rama de imagen (CNN)

### Curvas de pérdida de la CNN
**Archivo:** `rama_cnn/cnn_perdidas.png`

**Qué muestra:** la pérdida en train y en validation a lo largo de las 26
épocas, con una línea vertical que separa visualmente la Fase 1 (cabeza
congelada) de la Fase 2 (fine-tuning).

**Qué destacar:**
- La Fase 1 (épocas 1-8) muestra una mejora lenta porque solo la cabeza puede
  ajustarse: la capacidad de aprendizaje es limitada.
- Al iniciar la Fase 2 hay una bajada acusada de ambas pérdidas, prueba de que
  el fine-tuning del backbone aporta capacidad sustancial.
- Train y validation evolucionan en paralelo durante todo el entrenamiento, sin
  divergencia significativa. Esto indica que no hubo sobreajuste apreciable.
- El early stopping paró el entrenamiento por estabilización del val_loss, no
  por degradación. El modelo podría haber seguido aprendiendo lentamente.

---

### Macro-F1 en validation durante el entrenamiento
**Archivo:** `rama_cnn/cnn_macro_f1.png`

**Qué muestra:** la evolución del macro-F1 en validation a lo largo de las
26 épocas, con una estrella marcando el mejor punto (el modelo que se
guardó).

**Qué destacar:**
- En Fase 1 el macro-F1 se estanca alrededor de 0.42, reforzando la idea de
  que la cabeza por sí sola tiene capacidad limitada.
- En Fase 2 sube de forma sostenida hasta 0.59 en la época 20, lo que justifica
  la decisión de descongelar bloques del backbone.

---

### Matrices de confusión de la CNN
**Archivo:** `rama_cnn/confusion_matrices_cnn.png`

**Qué muestra:** una matriz de confusión 2×2 por clase, sobre test, con los
umbrales óptimos por clase aplicados.

**Qué destacar:**
- La CNN tiene un rendimiento claramente mejor que la rama tabular en Agarre
  (58 verdaderos positivos de 66 reales) y en Pierna (100 de 123).
- En Cabeza el rendimiento es similar al de la rama tabular (87 verdaderos
  positivos de 130).
- Distancia sigue siendo la clase más difícil, incluso para la CNN.

---

## Rama de fusión

### Matrices de confusión del modelo de fusión
**Archivo:** `rama_fusion/confusion_matrices_fusion.png`

**Qué muestra:** una matriz de confusión 2×2 por clase del modelo final
(fusión por stacking), sobre test.

**Qué destacar en el texto:**
- La fusión mejora o iguala el rendimiento en casi todas las clases respecto
  a las dos ramas individuales.
- Pierna alcanza un rendimiento muy bueno (104 verdaderos positivos de 123
  reales con baja tasa de falsos positivos).
- Distancia mejora respecto a la rama tabular y queda en línea con la CNN.

---

### Pesos óptimos de la fusión por clase
**Archivo:** `ramas_fusion/pesos_fusion.png`

**Qué muestra:** barras apiladas que descomponen el peso óptimo en la fusión
ponderada para cada clase, separando la contribución de la rama tabular y
la rama CNN.

**Qué destacar:**
- En Cabeza la rama tabular pesa solo 0.15, mientras que la CNN tiene 0.85.
  Cuando ambas ramas son competitivas en solitario en una clase, el modelo
  apuesta por la más fuerte y descarta la otra.
- En Agarre y Pierna el reparto está más equilibrado (0.45 / 0.55), señal de
  que las dos ramas aportan información complementaria.
- En clases donde la CNN domina (Distancia, Dorsal, Lumbar), la tabular sigue
  aportando entre 0.25 y 0.30 de peso: no es prescindible.

---

### Barrido del peso de fusión por clase
**Archivo:** `rama_fusion/fusion_barrido_pesos.png`

**Qué muestra:** para cada clase, una curva con el F1 óptimo en validation
en función del peso `w` de la rama tabular (de 0 a 1). Las estrellas marcan
el punto óptimo de cada curva.

**Qué destacar:**
- Cada clase tiene su pico en un valor de `w` distinto, lo que confirma que
  un peso fijo (como 0.5 del promedio simple) no es óptimo.
- Los extremos `w=0` y `w=1` corresponden a usar solo la CNN o solo la rama
  tabular respectivamente. Comparando los extremos se ve qué rama era más
  fuerte en solitario para cada clase.
- Las curvas son suaves: el F1 cambia gradualmente con `w`, indicando que la
  fusión es robusta frente a pequeñas variaciones del peso.

---

## Resultados globales y comparativas

### Comparativa de macro-F1 entre modelos
**Archivo:** `rama_fusion/macro_f1_modelos.png`

**Qué muestra:** gráfico de barras horizontal con el macro-F1 en test de los
cuatro modelos principales (Random Forest, XGBoost, CNN, fusión).

**Qué destacar:**
- La progresión es monotónica: RF (0.53) → XGBoost (0.55) → CNN (0.64) →
  fusión (0.65).
- El salto más grande es entre XGBoost y CNN (+0.09), lo que muestra el valor
  de la información visual fina.
- La fusión añade un pequeño pero consistente extra sobre la CNN, validando
  que las dos ramas son complementarias.

---

### F1 por clase: tabular, CNN y fusión
**Archivo:** `rama_fusion/f1_por_clase.png`

**Qué muestra:** gráfico de barras agrupadas. Una agrupación por clase, con
tres barras por agrupación correspondientes a las tres aproximaciones
finales.

**Qué destacar:**
- La CNN supera claramente a la tabular en Agarre (+0.20), Distancia (+0.17) y
  Pierna (+0.13). Son clases con señal en píxeles finos.
- En Cabeza la rama tabular es ligeramente superior a la CNN (+0.03), porque
  el ángulo cuello-tronco calculado capta bien esa información.
- La fusión iguala o supera a la mejor rama en casi todas las clases. La
  excepción notable es Agarre, donde la fusión queda ligeramente por debajo
  de la CNN sola (0.69 vs 0.73). Es honesto mencionarlo.

---

### Mejora de la fusión sobre cada rama individual
**Archivo:** `rama_fusion/mejora_fusion.png`

**Qué muestra:** gráfico de barras agrupadas con la diferencia de F1 entre la
fusión y cada rama por separado, por clase. Valores positivos indican que la
fusión mejora, negativos que empeora.

**Qué destacar:**
- Respecto a la rama tabular, la fusión mejora en todas las clases excepto
  marginalmente Pierna, con mejoras de hasta +0.16 (Agarre) y +0.13 (Distancia).
- Respecto a la CNN, la fusión mejora en Cabeza (+0.05), Dorsal (+0.03),
  Lumbar (+0.03) y Pierna (+0.04), e iguala o empeora levemente en Agarre y
  Distancia.
- Esto refuerza la conclusión: la fusión no es una mejora uniforme, sino una
  redistribución del rendimiento donde las debilidades de cada rama quedan
  compensadas por la otra.

---

## Resumen rápido por sección de la memoria/presentación (sugerencia)

| Sección | Gráficas recomendadas |
|---------|----------------------|
| Introducción | Distribución de etiquetas, verificación visual de pose |
| Pipeline general | Diagrama del pipeline (a mano) + verificación visual de pose |
| Dataset | Distribución de etiquetas, co-ocurrencia |
| Features | Poder discriminativo, correlación entre features |
| Rama tabular | Convergencia XGBoost, importancia features, matrices de confusión |
| Rama CNN | Curvas de pérdida, curva de macro-F1, matrices de confusión |
| Rama fusión | Pesos por clase, barrido de pesos, matrices de confusión |
| Resultados globales | Comparativa macro-F1, F1 por clase, mejora de fusión |
| Discusión | Co-ocurrencia, curvas precision-recall, mejora de fusión |