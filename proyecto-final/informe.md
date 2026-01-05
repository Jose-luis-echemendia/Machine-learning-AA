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

El **PAMAP2 Physical Activity Monitoring Dataset** es un conjunto de datos orientado al reconocimiento de actividades humanas (HAR). Fue recopilado mediante tres dispositivos portables _Inertial Measurement Units (IMU)_ colocados en el pecho, la mano y el tobillo, además de un pulsómetro que registró el ritmo cardíaco.

Incluye mediciones de aceleración tridimensional, giroscopio, magnetómetro, orientación, temperatura corporal y frecuencia cardíaca. La información fue capturada por **9 participantes** mientras realizaban **18 actividades físicas**, entre ellas: caminar, correr, descansar, limpiar, subir escaleras, ciclismo, entre otras. Esto convierte a PAMAP2 en un dataset complejo y adecuado para evaluar algoritmos de aprendizaje automático sobre señales fisiológicas y de movimiento.

### Problema a Resolver

El objetivo central de este trabajo es **clasificar la actividad física que una persona está realizando a partir de datos sensoriales multivariados**. El reconocimiento automático de actividades es un campo clave en aplicaciones como monitoreo de salud, entrenamiento deportivo, asistencia a adultos mayores y sistemas inteligentes.

El problema se aborda como una **tarea de clasificación multiclase** en la que cada instancia de entrada contiene decenas de señales fisiológicas y de movimiento, y la salida esperado es una de las 18 actividades registradas. Por tanto, corresponde a un problema de **aprendizaje supervisado**.

### Descripción de los Algoritmos a Utilizar

Para este estudio se seleccionan tres algoritmos de clasificación ampliamente utilizados en el procesamiento de datos sensoriales:

#### • K-Nearest Neighbors (KNN)

KNN es un algoritmo basado en distancias que clasifica una instancia en función de las etiquetas de sus vecinos más cercanos. Es simple, no requiere entrenamiento costoso y suele funcionar bien en datos sensoriales normalizados. Como desventaja, su rendimiento se degrada en datasets grandes debido al costo de búsqueda. En el contexto de señales IMU, KNN puede capturar patrones locales de movimiento al comparar instancias similares en el espacio de características.

#### • Árboles de Decisión

Los Árboles de Decisión son modelos jerárquicos que dividen el espacio de características mediante reglas de decisión basadas en umbrales. Son fáciles de interpretar, no requieren normalización de datos y pueden capturar relaciones no lineales. Sin embargo, tienden al sobreajuste si no se controla su profundidad. En el reconocimiento de actividades, pueden identificar patrones discriminativos como rangos específicos de aceleración o frecuencia cardíaca que caracterizan cada actividad física.

#### • Redes Neuronales

Las Redes Neuronales son modelos de aprendizaje profundo compuestos por capas de neuronas artificiales capaces de aprender representaciones complejas y no lineales de los datos. Son especialmente efectivas en datos secuenciales y multivariados como las señales IMU. Requieren mayor cantidad de datos y tiempo de entrenamiento, pero pueden lograr un rendimiento superior al capturar dependencias temporales y espaciales entre sensores. Para este trabajo se utilizará una arquitectura feedforward (perceptrón multicapa) que permita modelar las relaciones complejas entre las características sensoriales y las actividades físicas.

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

Listado de fuentes utilizadas, con formato APA/IEEE.
