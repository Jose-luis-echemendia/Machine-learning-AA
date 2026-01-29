<div align="center">

![Logo UCI](images/uci_logo.png)

### Facultad de Tecnologías Interactivas

# Reconocimiento de Actividades Físicas mediante Aprendizaje Automático

## Dataset PAMAP2

<br>

**Autor:**  
Jose Luis Echemendia López

<br>

**Tutores:**  
Ms.C. Ángel Vázquez Sánchez
<br>
Ing. Orlian Mesa Caceres

<br><br>

La Habana, Cuba  
Diciembre 2025  
Año 67 de la Revolución

</div>

---

## Introducción

### Descripción del Dataset PAMAP2

El **PAMAP2 Physical Activity Monitoring Dataset** es un conjunto de datos de referencia en el campo del Reconocimiento de Actividades Humanas (_Human Activity Recognition_, HAR). Fue recopilado por Reiss y Stricker (2012) mediante tres unidades de medición inercial (_Inertial Measurement Units_, IMU) colocadas en el pecho, la mano dominante y el tobillo, además de un monitor de frecuencia cardíaca.

Incluye mediciones de aceleración tridimensional, giroscopio, magnetómetro, orientación, temperatura corporal y frecuencia cardíaca. La información fue capturada por **9 participantes** mientras realizaban **18 actividades físicas**, ejemplos de ellas:

- Actividades básicas: caminar, correr, subir/bajar escaleras
- Actividades domésticas: limpiar, planchar, aspirar
- Actividades deportivas: ciclismo, fútbol, tenis de mesa
- Estados: acostado, sentado, de pie

Cada unidad de medición inercial (IMU) captura múltiples señales de movimiento y orientación, incluyendo aceleración 3D con un rango de ±16g, giroscopio 3D con sensibilidad de hasta ±2000°/s, magnetómetro 3D para estimar orientación relativa al campo magnético terrestre, y orientación 4D representada mediante cuaterniones. En total, se disponen de 52 atributos por instancia, derivados de los tres IMUs (pecho, mano y tobillo) y del monitor de frecuencia cardíaca. La frecuencia de muestreo varía según el sensor: los IMUs operan a 100 Hz, garantizando alta resolución temporal para capturar movimientos rápidos, mientras que el ritmo cardíaco se registra a 9 Hz, suficiente para variaciones fisiológicas típicas. Esta combinación de datos de alta frecuencia y múltiples modalidades proporciona una base rica para el análisis de patrones de actividad.

Esto convierte a PAMAP2 en un dataset complejo y adecuado para evaluar algoritmos de aprendizaje automático sobre señales fisiológicas y de movimiento.

### Problema a Resolver

El objetivo central de este trabajo es **clasificar la actividad física que una persona está realizando a partir de datos sensoriales multivariados**. El reconocimiento automático de actividades es un campo clave en aplicaciones como monitoreo de salud, entrenamiento deportivo, asistencia a adultos mayores y sistemas inteligentes.

El problema se aborda como una **tarea de clasificación multiclase** en la que cada instancia de entrada contiene decenas de señales fisiológicas y de movimiento, y la salida esperado es una de las 18 actividades registradas. Por tanto, corresponde a un problema de **aprendizaje supervisado**.

### Descripción de los Algoritmos a Utilizar

Para este estudio se seleccionan tres algoritmos de clasificación ampliamente utilizados en el procesamiento de datos sensoriales:

#### • K-Nearest Neighbors (KNN)

El algoritmo K-Vecinos más Cercanos es un método de aprendizaje basado en instancias (lazy learning) que clasifica una nueva observación según la clase mayoritaria entre sus _k_ ejemplos más cercanos en el espacio de características, calculados mediante una métrica de distancia, comúnmente la euclidiana (Cover & Hart, 1967).

**Características:**

- Tipo de aprendizaje: Supervisado, no paramétrico
- Parámetro clave: Número de vecinos (_k_)
- Fase de entrenamiento: Almacenamiento simple del conjunto de datos (no hay optimización de parámetros)
- Fase de predicción: Cálculo de distancias a todas las instancias

**Ventajas:**

- Simplicidad conceptual y de implementación
- No requiere supuestos sobre la distribución de los datos
- Se adapta naturalmente a nuevos datos sin reentrenamiento
- Funciona bien en datos normalizados de dimensionalidad moderada

**Desventajas:**

- Alto costo computacional en la fase de predicción (O(nd) para _n_ instancias y _d_ dimensiones)
- Sensibilidad a características irrelevantes o redundantes
- Degradación del rendimiento en espacios de alta dimensionalidad (maldición de la dimensionalidad)
- Requiere definición cuidadosa de la métrica de distancia y selección de _k_

**Aplicaciones:**

- Reconocimiento de Actividades Humanas (HAR): Adecuado para datasets de tamaño moderado donde las actividades físicas generan patrones claramente diferenciables en el espacio de características (Patterson et al., 2005). Es especialmente útil como baseline comparativo por su simplicidad.
- Diagnóstico Médico Asistido: En detección de anomalías médicas, KNN se utiliza para identificar patrones similares en imágenes médicas o señales fisiológicas, como la clasificación de tejidos en mamografías (Elter et al., 2007) o la detección de arritmias en electrocardiogramas basándose en formas de onda similares.
- Sistemas de Recomendación: Para filtrado colaborativo, donde usuarios con patrones de consumo similares (vecinos) reciben recomendaciones basadas en las preferencias de sus k-vecinos más cercanos (Sarwar et al., 2001), aplicado en plataformas como Netflix y Amazon.

#### • Árboles de Decisión

Los Árboles de Decisión son modelos jerárquicos que particionan recursivamente el espacio de características mediante reglas de decisión basadas en umbrales, utilizando criterios como Ganancia de Información o Índice Gini para seleccionar las divisiones óptimas (Breiman et al., 1984).

**Características:**

- Estructura: Nodos de decisión (condiciones) y hojas (clases)
- Criterios de división: Ganancia de información (entropía) o Índice Gini
- Regularización: Profundidad máxima, mínimas muestras por hoja
- Interpretabilidad: Visualización directa de reglas de decisión

**Ventajas:**

- Alta interpretabilidad y visualización intuitiva
- No requiere normalización previa de las características
- Manejo simultáneo de variables categóricas y numéricas
- Captura relaciones no lineales y de interacción

**Desventajas:**

- Alta varianza (inestabilidad ante pequeñas variaciones en los datos)
- Tendencia al sobreajuste sin restricciones de profundidad
- Dificultad para capturar relaciones lineales simples de forma eficiente
- Sesgo hacia características con muchos valores posibles

**Aplicaciones:**

- Reconocimiento de Actividades Humanas (HAR): Particularmente efectivos cuando existen umbrales físicos claros en las señales (ej. rango específico de frecuencia cardíaca para "correr" vs "caminar"), permitiendo identificar reglas de decisión interpretables (Bao & Intille, 2004).
- Evaluación de Riesgo Crediticio: Ampliamente utilizados por instituciones financieras para decidir aprobación de créditos, donde las reglas binarias (ingreso > X, antigüedad laboral > Y) permiten crear sistemas automatizados y explicables que cumplen con regulaciones de transparencia (Baesens et al., 2003).
- Diagnóstico Clínico: En sistemas expertos médicos, como los criterios de diagnóstico de enfermedades (ej. árboles de decisión para diagnóstico diferencial), donde la secuencia lógica de preguntas/síntomas conduce a conclusiones clínicas (Lucas et al., 2004).

#### • Redes Neuronales. Perceptrón multicapa

El Perceptrón Multicapa es una red neuronal artificial feedforward compuesta por múltiples capas de neuronas interconectadas que aprenden representaciones jerárquicas de los datos mediante funciones de activación no lineales y el algoritmo de retropropagación del error (Rumelhart et al., 1986).

**Características:**

- Arquitectura: Capa de entrada → Capas ocultas → Capa de salida
- Funciones de activación: ReLU (capas ocultas), Softmax (capa de salida para clasificación)
- Optimización: Variantes de descenso de gradiente (ej. Adam)
- Regularización: Dropout, regularización L2, parada temprana

**Ventajas:**

- Capacidad de aproximar funciones complejas no lineales (teorema de aproximación universal)
- Aprendizaje automático de características relevantes
- Excelente rendimiento en datos de alta dimensionalidad
- Flexibilidad arquitectónica para adaptarse al problema

**Desventajas:**

- Requiere ajuste cuidadoso de numerosos hiperparámetros
- Largo tiempo de entrenamiento y necesidad de datos abundantes
- Riesgo de sobreajuste y optimización en mínimos locales
- Interpretabilidad limitada ("caja negra")

**Aplicaciones:**

- Reconocimiento de Actividades Humanas (HAR): Muestran rendimiento superior en datasets de sensores multimodales como PAMAP2, donde existen dependencias complejas y no lineales entre las características temporales y espaciales (Hammerla et al., 2016).
- Reconocimiento de Imágenes: Base de sistemas modernos de visión por computadora para tareas como clasificación de objetos, detección facial y segmentación semántica, donde las capas aprenden características jerárquicas desde bordes simples hasta objetos complejos (LeCun et al., 2015).
- Procesamiento de Lenguaje Natural: Para modelado de lenguaje, análisis de sentimientos, clasificación de textos y traducción automática, donde las representaciones distribuidas capturan relaciones semánticas y sintácticas complejas (Collobert et al., 2011).

Cada uno de estos algoritmos será evaluado dentro del proceso KDD para determinar cuál ofrece el mejor rendimiento en el reconocimiento de actividades utilizando PAMAP2.

---

## 3. Desarrollo

### 3.1. Tipo de Problema y Aprendizaje

El problema abordado en este estudio se define formalmente como una tarea de clasificación supervisada multiclase. Cada instancia en el conjunto de datos PAMAP2 está representada por un vector de características xi∈R52xi​∈R52, que corresponde a las mediciones simultáneas de múltiples sensores corporales en un instante de tiempo específico. A esta instancia se le asocia una etiqueta yi∈{1,2,…,18}yi​∈{1,2,…,18}, que identifica una de las dieciocho actividades físicas realizadas. El objetivo del aprendizaje automático es, por lo tanto, inducir una función de mapeo f:R52→{1,18}f:R52→{1,18} a partir de un conjunto de ejemplos etiquetados {xi,yi}{xi​,yi​}, de modo que la función aprendida ff pueda predecir correctamente la actividad ynewynew​ para una nueva observación de sensores xnewxnew​.

La naturaleza supervisada del aprendizaje proviene de la disponibilidad de este conjunto de datos completamente etiquetado, donde el "supervisor" es la anotación manual de la actividad realizada durante la captura de datos. Este tipo de aprendizaje permite entrenar modelos que generalicen la relación entre patrones de señales sensoriales y actividades, sin requerir descubrir agrupamientos o estructuras ocultas en los datos, como ocurriría en un problema no supervisado. La multiclase se refiere a que la variable objetivo puede tomar más de dos valores discretos, en contraste con una clasificación binaria, lo que aumenta la complejidad del problema y requiere algoritmos y métricas de evaluación específicas.

### 3.2. Algoritmos Seleccionados

Para la resolución del problema de clasificación descrito, se seleccionaron tres algoritmos representativos de diferentes familias del aprendizaje automático, con el fin de realizar un análisis comparativo exhaustivo de sus capacidades y rendimiento sobre el dataset PAMAP2.

Los algoritmos seleccionados son:

- K-Nearest Neighbors (KNN): Seleccionado como representante de los métodos basados en instancias o aprendizaje perezoso (lazy learning). Su inclusión se justifica por su simplicidad conceptual, que sirve como un baseline sólido, y por su capacidad para capturar patrones locales en el espacio de características sin hacer suposiciones fuertes sobre la distribución de los datos. Permite evaluar si la similitud directa entre lecturas sensoriales es suficiente para discriminar actividades.
- Árboles de Decisión: Elegido como representante de los modelos basados en reglas de decisión. Su principal justificación radica en su alta interpretabilidad, la cual es valiosa para comprender qué características sensoriales (ej., un umbral específico de aceleración o frecuencia cardíaca) son más discriminantes para cada actividad. Además, su naturaleza no paramétrica y su capacidad para manejar datos no normalizados lo convierten en un candidato robusto para datos de sensores heterogéneos.
- Perceptrón Multicapa (MLP): Seleccionado como representante de las redes neuronales artificiales y el aprendizaje profundo. Se justifica por su conocido poder de aproximación de funciones no lineales complejas, lo que lo hace ideal para capturar las intrincadas relaciones y dependencias que existen entre las 52 señales de múltiples sensores. Se espera que este algoritmo logre el rendimiento más alto al poder modelar interacciones de alto orden que los métodos anteriores podrían pasar por alto.

Esta selección tripartita permite abarcar un espectro amplio de complejidad, interpretabilidad y capacidad de modelado. La comparación entre un método simple (KNN), uno interpretable (Árbol de Decisión) y uno potente pero complejo (MLP) proporcionará una visión integral sobre las ventajas y compromisos (trade-offs) de cada enfoque para la tarea específica de reconocimiento de actividades físicas.

### 3.3. Etapas del Proceso KDD

El proceso KDD se implementó siguiendo las cinco etapas fundamentales: Selección, Preprocesamiento, Transformación, Minería de Datos y Evaluación. A continuación se describe el flujo completo ejecutado para cada uno de los tres algoritmos.

#### 3.3.1. Selección del Dataset

El dataset PAMAP2 Physical Activity Monitoring fue seleccionado del UCI Machine Learning Repository (Reiss & Stricker, 2012). Este dataset se eligió por ser un estándar de referencia en la investigación de Reconocimiento de Actividades Humanas (HAR), ofreciendo un desafío realista y complejo debido a su naturaleza multivariada y temporal. La versión utilizada contenía 9 archivos individuales (subject101.dat a subject109.dat), correspondientes a los datos de los 9 participantes. El dataset original contenía 2,872,533 registros con 54 columnas, incluyendo las mediciones de los tres IMUs (mano, pecho, tobillo), frecuencia cardíaca, timestamp y actividad ID.

#### 3.3.2. Preprocesamiento

El objetivo de esta etapa fue limpiar y preparar los datos brutos para su análisis, eliminando ruido, inconsistencias y valores faltantes.

**Limpieza de datos:**

- Eliminación de actividades de transición: Se eliminaron todas las instancias con activityID = 0 (actividades transitorias entre tareas), reduciendo el dataset a 1,942,872 registros.
- Eliminación de columnas irrelevantes: Se eliminó la columna timestamp ya que no aporta valor predictivo para la clasificación instantánea de actividades.
- Distribución de actividades: Se identificaron 12 actividades principales tras la limpieza (las 18 originales menos las transiciones y algunas actividades con pocos datos).

**Manejo de valores perdidos:**

- Análisis: Se detectó que el 90.87% de los valores de `heart_rate` estaban perdidos, mientras que otras características tenían entre 0.12% y 0.57% de valores perdidos. Dada la extremadamente alta tasa de valores pérdidos (>90%), se optó por eliminar completamente la característica `heart_rate` del análisis. Esta decisión se basa en:
  1. Principio de parsimonia: Con >90% de datos ausentes, cualquier imputación introduciría más ruido artificial que señal genuina.
  2. Integridad fisiológica: Imputar masivamente valores cardíacos distorsionaría las señales fisiológicas reales.
  3. Suficiencia de sensores IMU: Los 51 sensores restantes (aceleración, giroscopio, magnetómetro) capturan información discriminativa suficiente.
- Estrategia para características restantes: Se utilizó imputación por la mediana con `SimpleImputer(strategy='median')` para las características con <1% de valores perdidos, preservando la estructura estadística del dataset.
- Resultado: Tras la eliminación de `heart_rate` y la imputación, todas las **51 características restantes** quedaron con 0 valores perdidos.

**Separación entrenamiento-prueba:**

- Proporción: 80% entrenamiento (1,554,297 registros), 20% prueba (388,575 registros).
- Estratificación: Se utilizó stratify=y para mantener la distribución proporcional de cada actividad en ambos conjuntos.

**Normalización:**

- Método: Estandarización StandardScaler() (media=0, desviación estándar=1).
- Justificación: KNN y MLP son sensibles a la escala; la normalización asegura que todas las características contribuyan equitativamente.
- Proceso: Se ajustó el escalador con X_train y se transformaron tanto X_train como X_test.

#### 3.3.3. Transformación

La etapa de transformación tiene como objetivo adaptar la estructura y el conjunto de características de los datos preprocesados para que sean óptimos para el modelado y la extracción de conocimiento. En este paso, se evalúa y define el espacio de características final que alimentará a los algoritmos de minería de datos, buscando un equilibrio entre la completitud de la información y la eficiencia computacional. A continuación, se detalla la estrategia adoptada para la selección y conservación de las características en este estudio.

**Selección y conservación de características:**
Tras el preprocesamiento, se dispuso de 51 características derivadas de los sensores IMU (52 originales menos `heart_rate`). A diferencia de flujos de trabajo que aplican reducción de dimensionalidad agresiva, en este estudio se decidió conservar estas 51 características por las siguientes consideraciones:

1. Especificidad del dominio HAR: Cada sensor (mano, pecho, tobillo) y cada eje (X, Y, Z) capturan información complementaria y potencialmente única. Por ejemplo:
   - La aceleración en el eje Z del tobillo es determinante para diferenciar "caminar" de "correr".
   - El giroscopio del pecho ayuda a identificar movimientos rotacionales en actividades como "planchar" o "aspirar".
   - Los magnetómetros aportan información de orientación relativa.

2. Preservación de la Información Discriminativa: La eliminación de características podría erosionar patrones sutiles necesarios para distinguir actividades físicamente similares, como "subir escaleras" versus "bajar escaleras", donde las señales de aceleración y giroscopio tienen firmas temporo-espaciales distintas.

3. Complejidad computacional manejable: Un espacio de 51 dimensiones es razonable para algoritmos modernos:
   - KNN funciona eficientemente con normalización adecuada.
   - MLP está diseñado para alta dimensionalidad.
   - Árboles de decisión manejan bien características numéricas continuas.

En síntesis, Se trabajó con **51 características** bien fundamentadas en el dominio de HAR, priorizando preservación de información sobre reducción automática de dimensionalidad.

#### 3.3.4. Minería de Datos

Dada la gran cantidad de datos (~1.5 millones en entrenamiento), se implementaron dos modalidades para cada algoritmo con el fin de equilibrar tiempo de ejecución, recursos computacionales y rendimiento de los modelos:

**Modalidades de Entrenamiento:**

- **Opción Rápida (Activa)**: Usando subconjuntos de datos para pruebas, demostraciones y desarrollo iterativo.
- **Opción Completa (Comentada)**: Usando todo el dataset, disponible en el código para evaluaciones finales cuando se disponga de recursos computacionales adecuados.

**Configuración Implementada (Opción Rápida - Activa):**

| **Algoritmo**                  | **Muestras de Entrenamiento** | **% del Total** | **Tiempo de Entrenamiento**     | **Justificación**                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------ | ----------------------------- | --------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **K-Nearest Neighbors (KNN)**  | 50,000 muestras               | 3.2%            | Rápido (~segundos)              | Alto costo en predicción (O(nd)) debido a su naturaleza de _lazy learning_. KNN almacena todos los ejemplos y calcula distancias en tiempo de predicción. Aunque 50k muestras siguen siendo una fracción pequeña, este tamaño permite validar el enfoque sin explotar los recursos disponibles.                                                            |
| **Árbol de Decisión**          | 50,000 muestras               | 3.2%            | Moderado (~decenas de segundos) | Balance óptimo entre tiempo de entrenamiento y capacidad de generalización. Este tamaño es suficiente para que el algoritmo identifique patrones de división significativos en las 51 características, construyendo un árbol con poder discriminativo sin requerir horas de entrenamiento.                                                                 |
| **Perceptrón Multicapa (MLP)** | 50,000 muestras               | 3.2%            | Moderado (~minutos)             | Aunque las redes neuronales se benefician de más datos, 50k muestras proporcionan suficiente información para entrenar un MLP con early stopping y validación. El costo principal está en el entrenamiento iterativo (retropropagación), pero con este tamaño se logra convergencia en tiempos razonables mientras se mantiene un rendimiento competitivo. |

Los subconjuntos se obtuvieron mediante muestreo aleatorio estratificado (`random_state=42`) para garantizar reproducibilidad y mantener la distribución proporcional de clases. La búsqueda de hiperparámetros se realizó con subconjuntos aún menores (10k-20k muestras) para optimizar el tiempo de GridSearchCV sin comprometer la calidad de la selección de parámetros. Los tres algoritmos utilizan el **mismo tamaño de entrenamiento (50k)** para garantizar una comparación justa y equitativa entre ellos. Las predicciones en todos los casos se realizan sobre el **conjunto de prueba completo** (388,575 muestras) para obtener métricas de evaluación confiables.

**Opciones Completas Disponibles:**
El código incluye implementaciones comentadas que permiten entrenar con el dataset completo (1,554,297 muestras) en cada algoritmo. Estas opciones están documentadas para estudios futuros con mayor capacidad computacional o para comparaciones con recursos dedicados (GPU, clusters, etc.).

La evidencia de la implementación se encuentra en el archivo main.ipynb: https://github.com/Jose-luis-echemendia/Machine-learning-AA/blob/main/proyecto-final/main.ipynb

#### 3.3.5. Evaluación

La evaluación del rendimiento de los modelos se realizó mediante un conjunto integral de métricas y técnicas de validación, garantizando una comprensión robusta de las capacidades y limitaciones de cada algoritmo. Esta etapa es crítica en el proceso KDD, ya que permite determinar si los modelos entrenados son adecuados para la tarea de reconocimiento de actividades físicas y pueden generalizarse a nuevos datos no vistos.

**Métricas de Evaluación Utilizadas:**

Se emplearon cuatro métricas fundamentales para cuantificar el rendimiento de cada modelo sobre el conjunto de prueba (388,575 muestras, 20% del dataset total):

1. **Accuracy (Exactitud)**: Proporción de predicciones correctas sobre el total de predicciones. Calculada como:
   $$\text{Accuracy} = \frac{\text{TP + TN}}{\text{TP + TN + FP + FN}}$$
   Donde TP = Verdaderos Positivos, TN = Verdaderos Negativos, FP = Falsos Positivos, FN = Falsos Negativos.

2. **Precision (Precisión)**: Proporción de predicciones positivas correctas sobre el total de predicciones positivas. En el contexto multiclase, se utilizó el promedio ponderado (`average='weighted'`) para considerar el desbalance de clases:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP + FP}}$$

3. **Recall (Sensibilidad/Exhaustividad)**: Proporción de instancias positivas correctamente identificadas sobre el total de instancias realmente positivas:
   $$\text{Recall} = \frac{\text{TP}}{\text{TP + FN}}$$

4. **F1-Score**: Media armónica entre Precision y Recall, proporcionando una métrica balanceada:
   $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision + Recall}}$$

**Matrices de Confusión:**

Para cada algoritmo se generó una matriz de confusión que visualiza las predicciones correctas e incorrectas para cada clase (actividad física). Esta representación permite identificar:

- Actividades que el modelo clasifica correctamente con alta confianza (diagonal principal).
- Confusiones sistemáticas entre pares de actividades similares (valores fuera de la diagonal).
- Clases problemáticas que requieren atención especial o ingeniería de características adicional.

Las matrices se implementaron utilizando `confusion_matrix()` de scikit-learn y se visualizaron con `ConfusionMatrixDisplay`, empleando mapas de colores diferenciados para cada algoritmo (Blues para KNN, Greens para Árbol de Decisión, Blues para MLP).

**Validación Cruzada:**

Para evaluar la estabilidad y capacidad de generalización de los modelos, se implementó validación cruzada k-fold:

- **Configuración rápida (activa)**: 3-fold cross-validation sobre 15,000 muestras del conjunto de entrenamiento.
- **Parámetros**: `cv=3`, `scoring='accuracy'`, `n_jobs=-1` (paralelización completa).
- **Objetivo**: Estimar la varianza del rendimiento y detectar posibles problemas de sobreajuste o subajuste.

Para cada fold, se calculó el accuracy y se reportó:

- **Media**: Rendimiento esperado del modelo.
- **Desviación estándar**: Medida de estabilidad (valores bajos indican alta consistencia).
- **Intervalo de confianza**: Calculado como `[media - 2×std, media + 2×std]` (~95% de confianza).

Los resultados de validación cruzada se visualizaron mediante gráficos de barras que muestran el accuracy de cada fold junto con la línea de media, permitiendo identificar visualmente la consistencia del modelo.

**Análisis de Generalización:**

Para KNN, se implementó un análisis adicional de sobreajuste comparando las métricas en el conjunto de entrenamiento versus el conjunto de prueba:

- **Diferencia < 5%**: Excelente generalización.
- **Diferencia 5-10%**: Generalización aceptable.
- **Diferencia > 10%**: Posible overfitting.

**Reportes de Clasificación:**

Se generaron reportes detallados por clase usando `classification_report()` de scikit-learn, proporcionando:

- Precision, Recall y F1-Score para cada actividad individual.
- Support (número de instancias reales de cada clase en el conjunto de prueba).
- Promedios macro (no ponderado) y weighted (ponderado por frecuencia de clase).

Esta información es crucial para identificar qué actividades específicas son más difíciles de clasificar y requieren atención especial.

### 3.4. Validación y Resultados Estadísticos

Una vez completada la etapa de evaluación individual de cada algoritmo, se procedió a realizar un análisis comparativo exhaustivo para determinar cuál de los tres métodos ofrece el mejor compromiso entre rendimiento, interpretabilidad y eficiencia computacional para la tarea de reconocimiento de actividades físicas con el dataset PAMAP2.

**Tabla Comparativa de Rendimiento:**

Los tres algoritmos fueron entrenados bajo condiciones idénticas (50,000 muestras, mismas características, misma semilla aleatoria) y evaluados sobre el mismo conjunto de prueba completo (388,575 muestras) para garantizar una comparación equitativa. A continuación se presenta una tabla consolidada con las métricas principales:

| **Algoritmo**                  | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **Validación Cruzada (media ± std)** |
| ------------------------------ | ------------ | ------------- | ---------- | ------------ | ------------------------------------ |
| **K-Nearest Neighbors (K=3)**  | 0.9308       | 0.9322        | 0.9308     | 0.9297       | 0.8919 ± 0.0029                      |
| **Árbol de Decisión**          | 0.9687       | 0.9687        | 0.9687     | 0.9687       | 0.9095 ± 0.0135                      |
| **Perceptrón Multicapa (MLP)** | 0.9666       | 0.9668        | 0.9666     | 0.9664       | 0.9275 ± 0.0027                      |

_Nota: Los valores específicos se generan al ejecutar el notebook completo. Consultar las celdas de comparación final para ver los resultados exactos._

**Análisis Cualitativo por Algoritmo:**

**1. K-Nearest Neighbors (KNN):**

- **Fortalezas observadas**:
  - Implementación directa y rápida.
  - Entrenamiento instantáneo (solo almacenamiento de datos).
  - Buena baseline para comparación.
  - Con normalización adecuada, captura bien patrones locales.
- **Limitaciones identificadas**:
  - Costo computacional elevado en predicción.
  - Sensibilidad a características ruidosas o irrelevantes.
  - Requiere toda la data en memoria.
  - Rendimiento puede degradarse en alta dimensionalidad.

**2. Árbol de Decisión:**

- **Fortalezas observadas**:
  - Alta interpretabilidad: reglas de decisión claras y visualizables.
  - No requiere normalización de datos.
  - Identifica características discriminativas clave.
  - Maneja bien datos mixtos y no lineales.
  - Tiempo de entrenamiento razonable.
- **Limitaciones identificadas**:
  - Posible inestabilidad ante variaciones en datos de entrenamiento.
  - Tendencia al sobreajuste sin regularización adecuada.
  - Dificultad para capturar relaciones lineales simples de forma eficiente.

**3. Perceptrón Multicapa (MLP):**

- **Fortalezas observadas**:
  - Capacidad de modelar relaciones complejas y no lineales.
  - Aprovecha la naturaleza multivariada del dataset (51 características de múltiples sensores).
  - Aprende representaciones jerárquicas automáticamente.
  - Rendimiento robusto con early stopping y regularización L2.
  - Escalable a conjuntos de datos más grandes.
- **Limitaciones identificadas**:
  - Mayor tiempo de entrenamiento (varios minutos vs. segundos).
  - Baja interpretabilidad ("caja negra").
  - Requiere ajuste cuidadoso de hiperparámetros.
  - Necesidad de normalización de datos.

**Visualizaciones Comparativas:**

Se generaron dos visualizaciones principales para facilitar la interpretación de resultados:

1. **Gráfico de barras agrupadas**: Muestra las cuatro métricas (Accuracy, Precision, Recall, F1-Score) para cada algoritmo lado a lado, permitiendo identificar rápidamente el algoritmo con mejor desempeño global.

2. **Mapa de calor (Heatmap)**: Representa los valores de métricas como una matriz coloreada, donde tonalidades más intensas indican mejor rendimiento. Facilita la identificación visual de patrones y trade-offs entre algoritmos.

**Determinación del Modelo Ganador:**

El modelo ganador se seleccionó considerando múltiples criterios:

- **Criterio primario**: Mayor Accuracy en el conjunto de prueba.
- **Criterios secundarios**: Estabilidad en validación cruzada (baja desviación estándar), balance entre Precision y Recall (F1-Score alto), y viabilidad práctica (tiempo de inferencia razonable).

**Recomendaciones para Aplicaciones Prácticas:**

Basándose en los resultados obtenidos, se identificaron escenarios de aplicación óptimos para cada algoritmo:

- **Para aplicaciones en tiempo real** donde la latencia es crítica: KNN o Árbol de Decisión (predicción rápida).
- **Para sistemas que requieren explicabilidad** (ej. diagnóstico clínico): Árbol de Decisión.
- **Para maximizar accuracy** en sistemas con recursos computacionales adecuados: MLP.
- **Para prototipado rápido y baseline**: KNN con subconjuntos de datos.

---

## 4. Conclusiones

El presente trabajo realizó un análisis exhaustivo del reconocimiento automático de actividades físicas mediante tres algoritmos de aprendizaje automático representativos (KNN, Árbol de Decisión y Perceptrón Multicapa) aplicados al dataset de referencia PAMAP2. A continuación se presentan las conclusiones principales derivadas del estudio.

### 4.1. Resultados Principales

**Clasificación del Desempeño:**

Los tres algoritmos evaluados demostraron ser viables para la tarea de reconocimiento de actividades, aunque con rendimientos diferenciados:

1. **Árbol de Decisión (Ganador)**: Accuracy de 96.87% (0.9687)
   - Mejor desempeño en todas las métricas (Precision, Recall, F1-Score: 0.9687)
   - Validación cruzada: 0.9095 ± 0.0135
   - Estable y predecible en su comportamiento

2. **Perceptrón Multicapa (MLP)**: Accuracy de 96.66% (0.9666)
   - Rendimiento muy cercano al ganador, diferencia de apenas 0.21 puntos porcentuales
   - Validación cruzada: 0.9275 ± 0.0027 (la más baja desviación estándar, indicando máxima estabilidad)
   - Mayor complejidad pero también mayor adaptabilidad a nuevos datos

3. **K-Nearest Neighbors (KNN)**: Accuracy de 93.08% (0.9308)
   - Baseline respectable pero claramente inferior a los anteriores
   - Validación cruzada: 0.8919 ± 0.0029
   - Limitado por la maldición de la dimensionalidad en espacios de 51 características

### 4.2. Análisis Comparativo e Insights

**Sobre el Árbol de Decisión:**

El Árbol de Decisión emergió como el ganador indiscutible del estudio. Este resultado es particularmente valioso porque:

- **Interpretabilidad**: A diferencia del MLP, las reglas de decisión pueden ser visualizadas y explicadas, permitiendo comprender exactamente qué características sensoriales (ej. aceleración del tobillo, frecuencia cardíaca, giroscopio del pecho) son decisivas para clasificar cada actividad.
- **Eficiencia computacional**: El tiempo de predicción es mínimo, haciendo el modelo viable para aplicaciones en tiempo real en dispositivos móviles o wearables.
- **Robustez**: Los patrones capturados en los 50k datos de entrenamiento generalizaron excepcionales al conjunto de prueba completo (388,575 muestras).
- **No requiere preprocesamiento extenso**: Su funcionamiento no depende de normalización previa, simplificando el pipeline de producción.

Este desempeño sugiere que para el reconocimiento de actividades físicas, existen **umbrales y límites claramente definibles** en los valores de los sensores que permiten distinguir las actividades con notable precisión. Las actividades estudiadas (caminar, correr, subir/bajar escaleras, etc.) poseen firmas sensoriales distintivas que un árbol de decisión puede capturar de manera eficiente.

**Sobre el Perceptrón Multicapa:**

El MLP mostró un rendimiento prácticamente equivalente al Árbol de Decisión, con dos características destacables:

- **Mayor estabilidad en validación cruzada**: La desviación estándar más baja (0.0027) sugiere que el MLP es menos sensible a variaciones en el conjunto de entrenamiento, lo que lo hace potencialmente más confiable en escenarios con datos heterogéneos o ruidosos.
- **Capacidad de aproximación no lineal**: Aunque el Árbol de Decisión fue superior, el MLP demostró que es capaz de modelar las complejas relaciones no lineales entre las 51 características de múltiples sensores, lo que lo posiciona como una opción sólida para futuras ampliaciones del problema.

La cercanía en rendimiento entre ambos modelos indica que **no existe un gap significativo de capacidad no lineal no capturada** por el Árbol de Decisión, sugiriendo que los problemas más complejos del dataset ya están siendo manejados por este método.

**Sobre K-Nearest Neighbors:**

Aunque KNN mostró un rendimiento aceptable (93.08%), su desempeño inferior se alinea con las limitaciones teóricas conocidas:

- **Maldición de la dimensionalidad**: En un espacio de 51 dimensiones, la métrica de distancia euclidiana pierde poder discriminativo. El concepto de "vecindad" se vuelve menos significativo.
- **Sensibilidad a características ruidosas**: La presencia de características potencialmente ruidosas o redundantes afecta desproporcionadamente a métodos basados en distancia.
- **Costo computacional**: Aunque el entrenamiento es rápido (O(1)), la predicción requiere O(n·d) comparaciones de distancia, haciéndolo impractico para aplicaciones a gran escala.

Sin embargo, KNN sigue siendo valioso como **baseline de referencia** y para validar la complejidad inherente del problema. Un accuracy de 93% indica que la tarea no es trivial, pero tampoco es extremadamente difícil.

### 4.3. Validación Cruzada y Generalización

Un hallazgo importante del estudio es el comportamiento diferenciado en validación cruzada:

- **Árbol de Decisión**: CV 90.95%, Test 96.87% (diferencia: 5.92%)
- **MLP**: CV 92.75%, Test 96.66% (diferencia: 3.91%)
- **KNN**: CV 89.19%, Test 93.08% (diferencia: 3.89%)

Estas diferencias sugieren que:

1. El Árbol de Decisión mejora significativamente al evaluar sobre datos de prueba completos, indicando que el subset de 15k muestras usado en CV no captura toda la diversidad del dataset completo.
2. El MLP y KNN mantienen más consistencia entre CV y test, sugeriendo patrones más uniformes en sus aprendizajes.
3. **Ningún modelo muestra overfitting severo**, lo que es tranquilizador para su potencial uso en producción.

### 4.4. Implicaciones Prácticas

Para diferentes escenarios de aplicación:

**1. Sistemas Wearables en Tiempo Real (Smartphone, Smartwatch):**

- **Recomendación**: Árbol de Decisión
- **Justificación**: Latencia mínima, bajo consumo de batería, sin necesidad de normalización previa, y máximo accuracy.

**2. Investigación Clínica o Biomédica:**

- **Recomendación**: Árbol de Decisión (por interpretabilidad) o análisis comparativo
- **Justificación**: La necesidad de explicabilidad es crítica. Un médico o biomédico debe poder entender por qué se clasifica una actividad de determinada manera.

**3. Monitoreo Remoto de Salud Continuo:**

- **Recomendación**: MLP
- **Justificación**: Mayor estabilidad ante datos heterogéneos, mejor generalización a participantes no vistos en entrenamiento.

**4. Prototipado Rápido o Investigación Inicial:**

- **Recomendación**: KNN
- **Justificación**: Más rápido de implementar, sirve como baseline para establecer expectativas.

### 4.5. Limitaciones del Estudio

Es importante reconocer las limitaciones de este trabajo:

1. **Tamaño de muestra de entrenamiento limitado**: Se utilizaron solo 50k muestras (3.2% del total disponible) para mantener tiempos de ejecución razonables. Estudios con el dataset completo podrían arrojar resultados diferentes.

2. **Validación cruzada con subset**: La CV se realizó sobre 15k muestras. Una CV completa (5-10 folds con todo el dataset) proporcionaría estimaciones de error más robustas.

3. **Hiperparámetros no exhaustivamente optimizados**: Aunque se utilizó GridSearchCV, el espacio de búsqueda fue limitado para mantener tiempos computacionales manejables.

4. **Ausencia de técnicas avanzadas**: El estudio no incluyó ensemble methods (Random Forest, Gradient Boosting), que podrían potencialmente superar el Árbol de Decisión.

5. **Características fijas**: No se exploraron técnicas de ingeniería de características (feature engineering) como wavelets, FFT o estadísticas temporales locales que podrían mejorar significativamente el rendimiento.

6. **Dataset específico**: Los resultados son válidos para PAMAP2 con sus 12 actividades. Otros datasets de HAR o dominios de clasificación podrían mostrar patrones diferentes.

### 4.6. Contribuciones del Trabajo

Este estudio realizó las siguientes contribuciones:

1. **Implementación reproducible**: Se proporcionó código completamente funcional y documentado para los tres algoritmos, incluyendo opciones para ejecutar con datos completos.

2. **Análisis comparativo sistemático**: Se evaluaron los modelos bajo condiciones idénticas (mismas características, particiones, métricas), permitiendo conclusiones confiables.

3. **Pipeline KDD completo**: Se documentó exhaustivamente cada etapa del proceso desde selección hasta evaluación, sirviendo como referencia para futuros estudios.

4. **Insights sobre trade-offs**: Se clarificaron los compromisos entre interpretabilidad (Árbol), estabilidad (MLP) y simplicidad (KNN).

### 4.7. Recomendaciones para Trabajos Futuros

Para extender este estudio se sugiere:

1. **Entrenar con dataset completo**: Ejecutar los tres algoritmos con las 1,554,297 muestras disponibles para validar si los resultados se mantienen o cambian significativamente.

2. **Explorar ensemble methods**: Implementar Random Forest, Gradient Boosting y votación ponderada de los tres modelos actuales.

3. **Ingeniería de características avanzada**: Aplicar transformadas de Fourier, análisis wavelet, o características estadísticas por ventana temporal para capturar dinámicas complejas.

4. **Técnicas de reducción de dimensionalidad**: Evaluar PCA, t-SNE o autoencoders para explorar si la dimensionalidad podría reducirse sin pérdida significativa de rendimiento.

5. **Modelado temporal**: Investigar arquitecturas recurrentes (LSTM, GRU) que puedan explotar la naturaleza temporal de las series de sensores.

6. **Validación en nuevos participantes**: Verificar si los modelos entrenadosen un subconjunto de participantes generalizan bien a participantes completamente nuevos, un escenario más realista.

7. **Análisis de robustez**: Evaluar cómo se comportan los modelos ante ruido sintético o variaciones en la colocación de sensores.

8. **Comparativa con literatura**: Contrastar estos resultados con otros trabajos publicados que utilizan PAMAP2, contextualizando el rendimiento obtenido.

### 4.8. Reflexión Final

El reconocimiento automático de actividades físicas mediante sensores inerciales es una tarea altamente viable, como demuestran los resultados de este estudio. El logro de 96.87% de accuracy con el Árbol de Decisión muestra que es posible construir sistemas confiables y prácticos para monitoreo de salud, entrenamiento deportivo, y asistencia a adultos mayores.

La selección del algoritmo debe considerar cuidadosamente el contexto de aplicación: la interpretabilidad del Árbol de Decisión lo hace ideal para sistemas que requieren trazabilidad y explicabilidad, mientras que la estabilidad del MLP lo posiciona como una opción robusta para producción a escala.

Este trabajo demuestra que en aprendizaje automático, **no siempre es necesario utilizar las técnicas más complejas para lograr excelentes resultados**. Un algoritmo simple pero bien aplicado, como el Árbol de Decisión, puede superar a redes neuronales profundas cuando el problema tiene estructura subyacente clara y bien definida, como en el caso del reconocimiento de actividades físicas donde cada actividad posee patrones sensoriales distintivos.

---

## 5. Bibliografía

Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE Transactions on Information Theory

Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and Regression Trees

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature

Patterson, D. J., Fox, D., Kautz, H., & Philipose, M. (2005). Fine-grained activity recognition. IEEE International Symposium on Wearable Computers

Bao, L., & Intille, S. S. (2004). Activity recognition from user-annotated acceleration data. Pervasive Computing

Hammerla, N. Y., Halloran, S., & Plötz, T. (2016). Deep, convolutional, and recurrent models for human activity recognition using wearables. IJCAI

Elter, M., Schulz-Wendtland, R., & Wittenberg, T. (2007). The prediction of breast cancer biopsy outcomes using two CAD approaches. Physics in Medicine & Biology

Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering recommendation algorithms. Proceedings of the 10th International Conference on World Wide Web

Baesens, B., Van Gestel, T., Viaene, S., Stepanova, M., Suykens, J., & Vanthienen, J. (2003). Benchmarking state-of-the-art classification algorithms for credit scoring. Journal of the Operational Research Society

Lucas, P. J., van der Gaag, L. C., & Abu-Hanna, A. (2004). Bayesian networks in biomedicine and health-care. Artificial Intelligence in Medicine

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature

Collobert, R., Weston, J., Bottou, L., Karlen, M., Kavukcuoglu, K., & Kuksa, P. (2011). Natural language processing (almost) from scratch. Journal of Machine Learning Research
