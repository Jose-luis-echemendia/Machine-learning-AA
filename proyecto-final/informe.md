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

Dada la gran cantidad de datos (~1.5 millones en entrenamiento), se implementaron dos modalidades para cada algoritmo:
- Opción Rápida: Usando subconjuntos de datos para pruebas y demostraciones.
- Opción Completa: Usando todo el dataset (comentada en el código para referencia).

| **Algoritmo** | **Muestras de Entrenamiento** | **% del Total** | **Tiempo de Entrenamiento** | **Justificación** |
|---------------|------------------------------|-----------------|-----------------------------|-------------------|
| **K-Nearest Neighbors (KNN)** | 20,000 muestras | 1.3% | Rápido (~segundos) | Alto costo en predicción (O(nd)) debido a su naturaleza de *lazy learning*. Prohibitivo usar dataset completo. |
| **Árbol de Decisión** | 50,000 muestras | 3.2% | Moderado (~decenas de segundos) | Balance entre tiempo de entrenamiento y capacidad de generalización. Suficiente para estimar reglas de división. |
| **Perceptrón Multicapa (MLP)** | **1,554,297 muestras** (completo) | **100%** | Lento (~750 segundos) | Las redes neuronales se benefician significativamente de más datos. Costo principal en entrenamiento, no en predicción. |

La evidencia de la implementación se encuentra en el archivo main.piynb: https://github.com/Jose-luis-echemendia/Machine-learning-AA/blob/main/proyecto-final/main.ipynb

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
