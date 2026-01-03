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

Interpretación de los resultados y análisis comparativo entre algoritmos.

---

## 5. Bibliografía

Listado de fuentes utilizadas, con formato APA/IEEE.
