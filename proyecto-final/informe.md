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
Ms.C. Ángel Vázquez
<br>
Ing. Orlian

<br><br>

La Habana, Cuba  
Diciembre 2025  
Año 67 de la Revolución

</div>

---

## Introducción

### Descripción del Dataset PAMAP2

El **PAMAP2 Physical Activity Monitoring Dataset**  es un conjunto de datos de referencia en el campo del Reconocimiento de Actividades Humanas (_Human Activity Recognition_, HAR). Fue recopilado por Reiss y Stricker (2012) mediante tres unidades de medición inercial (_Inertial Measurement Units_, IMU) colocadas en el pecho, la mano dominante y el tobillo, además de un monitor de frecuencia cardíaca.

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

El algoritmo K-Vecinos más Cercanos es un método de aprendizaje basado en instancias (lazy learning) que clasifica una nueva observación según la clase mayoritaria entre sus *k* ejemplos más cercanos en el espacio de características, calculados mediante una métrica de distancia, comúnmente la euclidiana (Cover & Hart, 1967).

**Características:**
- Tipo de aprendizaje: Supervisado, no paramétrico
- Parámetro clave: Número de vecinos (*k*)
- Fase de entrenamiento: Almacenamiento simple del conjunto de datos (no hay optimización de parámetros)
- Fase de predicción: Cálculo de distancias a todas las instancias

**Ventajas:**
- Simplicidad conceptual y de implementación
- No requiere supuestos sobre la distribución de los datos
- Se adapta naturalmente a nuevos datos sin reentrenamiento
- Funciona bien en datos normalizados de dimensionalidad moderada

**Desventajas:**
- Alto costo computacional en la fase de predicción (O(nd) para *n* instancias y *d* dimensiones)
- Sensibilidad a características irrelevantes o redundantes
- Degradación del rendimiento en espacios de alta dimensionalidad (maldición de la dimensionalidad)
- Requiere definición cuidadosa de la métrica de distancia y selección de *k*

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

## 3. Desarrollo – Proceso KDD

### 3.1. Tipo de Problema y Aprendizaje

Clasificación multiclase supervisada.

### 3.2. Algoritmos Seleccionados

Listado de los tres algoritmos escogidos y justificación.

### 3.3. Etapas del Proceso KDD

#### 3.3.1. Selección del Dataset

Descripción detallada.

#### 3.3.2. Preprocesamiento

- Limpieza de datos
- Manejo de valores perdidos
- Normalización

#### 3.3.3. Transformación

- Selección de características
- Ventaneo temporal (opc.)
- Reducción de dimensionalidad (opc.)

#### 3.3.4. Minería de Datos

Entrenamiento con:

- KNN (K-Nearest Neighbors)
- Árboles de Decisión
- Redes Neuronales (Perceptrón Multicapa)

Incluye parámetros, gráficas y explicación.

#### 3.3.5. Evaluación

- Matrices de confusión
- Accuracy, Precision, Recall, F1-score
- Cross-validation

### 3.4. Validación y Resultados Estadísticos

Tabla comparativa entre algoritmos.

### 3.5. Enlace a Google Colab

Insertar el link al cuaderno con el código.

---

## 4. Conclusiones

El algoritmo K-Nearest Neighbors (KNN) demostró un desempeño sobresaliente en la tarea de reconocimiento de actividades físicas utilizando el dataset PAMAP2. Con una configuración de K = 3, distancia euclidiana y un conjunto de 52 características sensoriales normalizadas, el modelo alcanzó un accuracy del 92.77% y un F1-score del 92.64% en el conjunto de prueba, lo que evidencia una alta capacidad discriminativa entre las actividades físicas analizadas.

El equilibrio observado entre precision (92.92%) y recall (92.77%) indica que el modelo no solo clasifica correctamente un alto porcentaje de instancias, sino que además mantiene un bajo nivel de falsos negativos, aspecto fundamental en aplicaciones de monitoreo de actividad física y salud. Este comportamiento sugiere que KNN logra capturar de manera efectiva patrones locales en el espacio de características, propios de las señales IMU y fisiológicas del dataset PAMAP2.

La validación cruzada 5-fold arrojó un accuracy promedio de 88.76% con una desviación estándar de ±0.26%, lo que refleja una buena estabilidad y capacidad de generalización del modelo. Aunque se observa una ligera disminución del rendimiento respecto al conjunto de prueba, esta diferencia es esperable y no indica sobreajuste significativo, sino más bien un comportamiento consistente frente a diferentes particiones de los datos.

El valor K = 3 resultó ser un compromiso adecuado entre sensibilidad al ruido y capacidad de generalización. Valores mayores de K podrían diluir la influencia de vecinos cercanos relevantes, mientras que valores menores aumentarían la sensibilidad a instancias atípicas. En este contexto, la elección de K permitió mantener una alta exactitud sin introducir ruido innecesario en la clasificación.

En conclusión, KNN se consolida como una solución sólida y confiable para el reconocimiento de actividades físicas en PAMAP2, especialmente en escenarios donde los datos están bien preprocesados y normalizados. Si bien su costo computacional puede ser una limitación en datasets de gran escala, los resultados obtenidos lo posicionan como un baseline robusto. Futuras mejoras podrían lograrse mediante el ajuste fino de K, el uso de ponderación por distancia o técnicas adicionales de selección de características, aunque el rendimiento actual ya puede considerarse excelente según las métricas evaluadas.

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