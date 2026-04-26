# AI Data Engineer Final Project

## Descripción

Este proyecto implementa un pipeline de visión computacional y procesamiento batch orientado a ingeniería de datos. El sistema detecta objetos en imágenes y videos utilizando el modelo YOLOv8n, genera un conjunto de detecciones enriquecidas en formato CSV, aplica un proceso ETL batch incremental para limpiar y transformar los datos a formato Parquet, y prepara la información para su posterior carga y análisis en Apache Hive.

El desarrollo fue realizado siguiendo los lineamientos del proyecto final de AI Data Engineer, separando claramente:

1. un sistema de clasificación de objetos, y
2. un sistema batch ETL construido en Python.

---

## Objetivos del proyecto

- Detectar objetos presentes en imágenes y videos.
- Extraer atributos relevantes de cada detección.
- Construir un dataset estructurado a partir de las detecciones.
- Aplicar transformaciones de limpieza y validación sobre los datos.
- Evitar duplicación de registros en ejecuciones repetidas del pipeline.
- Generar salidas analíticas compatibles con Apache Hive.

---

## Tecnologías utilizadas

- Python 3.10.20
- YOLOv8n
- Pandas
- NumPy
- OpenCV
- PyArrow
- Apache Hive
- Hadoop
- Java
- Makefile

---

## Requisitos del entorno

Para ejecutar este proyecto se requiere:

- Python 3.10.20
- Java
- Hadoop
- Hive
- Linux

Se recomienda trabajar dentro de un entorno virtual para aislar las dependencias del proyecto.

---

## Estructura del repositorio

```text
.
├── Makefile
├── requirements.txt
├── .gitignore
├── codigo
│   ├── batch_etl
│   │   ├── __init__.py
│   │   ├── processed_ids.txt
│   │   └── sistema_batch_etl.py
│   ├── clasificacion
│   │   ├── __init__.py
│   │   └── clasificacion.py
│   ├── csv
│   │   ├── batch_etl
│   │   └── clasificacion
│   ├── data
│   └── sql
│       ├── consultas_hive.sql
│       └── hive_tables.sql
├── tests
│   ├── __init__.py
│   ├── test_clasificacion.py
│   ├── test_etl.py
│   └── test_files.py
└── metastore_db
```

## 7. Datos de entrada

Los datos de entrada del proyecto se encuentran en:

```bash
codigo/data
```

El conjunto de datos utilizado incluye:

- al menos 20 imágenes,
- al menos 2 videos.

Tanto las imágenes como los videos utilizados en este proyecto fueron capturados personalmente con un **iPhone 16**, cumpliendo con el requisito de trabajar con material propio para la construcción del dataset.

### Formatos soportados

**Imágenes**
- `.jpg`
- `.jpeg`
- `.png`

**Videos**
- `.mp4`
- `.avi`
- `.mov`

Los archivos se procesan automáticamente por el sistema de clasificación.

### Imágenes

Las fotografías fueron tomadas personalmente con un **iPhone 16** y almacenadas dentro del conjunto de datos del proyecto. Estas imágenes representan escenas reales capturadas por el autor y fueron utilizadas como entrada para el proceso de detección de objetos.

### Videos

Los videos también fueron grabados personalmente con un **iPhone 16** y forman parte del conjunto de entrada del sistema. Estos archivos se utilizan para extraer detecciones por frame y generar información temporal agregada, incluyendo la columna `window_10s`.

---

## Ejecución del proyecto

El proyecto está automatizado mediante `Makefile`.

### Ejecutar solo la clasificación

```bash
make run-clasificacion
```

Este comando ejecuta el sistema de detección de objetos sobre los archivos de entrada y genera el archivo CSV de detecciones.

### Ejecutar solo el ETL batch

```bash
make run-etl
```

Este comando toma como entrada el archivo `detections.csv`, realiza la limpieza, validación y transformación de los datos, y genera la salida procesada en formato Parquet.

### Ejecutar todo el pipeline

```bash
make run-all
```

Este comando ejecuta el pipeline completo en el siguiente orden:

1. Clasificación
2. ETL batch
3. Inicialización SQL para Hive

---

## Comandos adicionales del Makefile

### Inicializar SQL para Hive

```bash
make sql-init
```

### Ejecutar pruebas

```bash
make test
```

### Ejecutar linting

```bash
make lint
```

### Formatear el código

```bash
make format
```

### Limpiar archivos temporales y basura

```bash
make clean
```

Este comando elimina archivos temporales, cachés, logs y archivos basura de Windows como `:Zone.Identifier`.

### Ver ayuda

```bash
make help
```

Este comando muestra los objetivos disponibles del `Makefile`.

---

## Sistema de clasificación

El sistema de clasificación está implementado en:

```bash
codigo/clasificacion/clasificacion.py
```

### Funcionalidad

Este módulo:

- procesa imágenes y videos automáticamente,
- utiliza el modelo preentrenado **YOLOv8n**,
- detecta objetos en cada imagen o frame de video,
- extrae información estructurada por cada detección,
- consolida los resultados en un archivo CSV.

### Salida del clasificador

La salida principal del sistema de clasificación se almacena en:

```bash
codigo/csv/clasificacion/detections.csv
```

---

## Esquema de datos generado

Cada detección contiene los siguientes atributos:

- `source_type`
- `source_id`
- `frame_number`
- `class_id`
- `class_name`
- `confidence`
- `x_min`
- `y_min`
- `x_max`
- `y_max`
- `width`
- `height`
- `area_pixels`
- `frame_width`
- `frame_height`
- `bbox_area_ratio`
- `center_x`
- `center_y`
- `center_x_norm`
- `center_y_norm`
- `position_region`
- `dominant_color_name`
- `dom_r`
- `dom_g`
- `dom_b`
- `timestamp_sec`
- `ingestion_date`
- `detection_id`

Estos campos permiten capturar información espacial, temporal y descriptiva de cada objeto detectado.

---

## Proceso ETL batch

El sistema ETL está implementado en:

```bash
codigo/batch_etl/sistema_batch_etl.py
```

### Entrada del ETL

El proceso toma como entrada el archivo generado por clasificación:

```bash
codigo/csv/clasificacion/detections.csv
```

### Transformaciones realizadas

Durante la ejecución del ETL se aplican las siguientes reglas:

- eliminación de registros con valores nulos,
- eliminación de confidencias fuera del rango válido `[0,1]`,
- eliminación de bounding boxes corruptos donde:
  - `x_max <= x_min`
  - `y_max <= y_min`
- normalización de algunos datos,
- creación de la columna `window_10s`,
- eliminación de duplicados usando `detection_id`.

### Salida del ETL

La salida del proceso ETL se genera en formato Parquet en la carpeta:

```bash
codigo/csv/batch_etl
```

---

## Estrategia de deduplicación e incrementalidad

El pipeline batch implementa una estrategia de checkpoint basada en el archivo:

```bash
codigo/batch_etl/processed_ids.txt
```

### Funcionamiento

- Si el archivo existe, se leen todos los `detection_id` previamente procesados.
- Estos identificadores se almacenan en un `set` para consulta eficiente.
- Si el archivo no existe, se inicializa un conjunto vacío.
- Solo se procesan registros nuevos.
- Si el ETL se ejecuta dos veces sobre los mismos datos, no se generan cambios ni archivos adicionales.

Esta estrategia asegura incrementalidad e idempotencia del pipeline.

---

## Integración con Hive

Los scripts SQL del proyecto se encuentran en:

```bash
codigo/sql/hive_tables.sql
codigo/sql/consultas_hive.sql
```

### Tabla principal

La tabla `detections` está diseñada para almacenar la salida procesada y permitir consultas analíticas sobre:

- clases detectadas,
- tipo de fuente,
- posición de objetos,
- ventanas temporales,
- color dominante,
- métricas espaciales.

---

## Flujo general del pipeline

1. Se colocan las imágenes y videos en `codigo/data`.
2. El módulo de clasificación detecta objetos y genera `detections.csv`.
3. El módulo ETL limpia, transforma y deduplica los datos.
4. La salida transformada se exporta en formato Parquet.
5. Los scripts SQL permiten estructurar y consultar la información en Hive.

---

## Pruebas

El proyecto incluye una carpeta `tests` con pruebas para distintos componentes:

- `test_clasificacion.py`
- `test_etl.py`
- `test_files.py`

Estas pruebas permiten validar la estructura general del pipeline, el comportamiento del sistema de clasificación y el procesamiento ETL.

Para ejecutar las pruebas:

```bash
make test
```

---

## Instalación de dependencias

Para instalar manualmente las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

Si el `Makefile` incluye objetivos de instalación, también puedes utilizar esos comandos.

---

## Consultas analíticas

Las consultas Hive del proyecto se encuentran en:

```bash
codigo/sql/consultas_hive.sql
```

Estas consultas permiten realizar análisis como:

- conteo de objetos por clase,
- detecciones por fuente,
- análisis por ventanas de 10 segundos,
- distribución por región dentro del frame,
- análisis de color dominante por clase.

---

## Consideraciones del proyecto

Este proyecto fue desarrollado siguiendo un enfoque de ingeniería de datos aplicada a visión computacional, cumpliendo con los siguientes principios:

- separación entre inferencia y ETL,
- almacenamiento intermedio en CSV,
- procesamiento batch en Python,
- transformación a Parquet,
- preparación para análisis en Hive,
- automatización mediante `Makefile`,
- control de duplicados mediante checkpoint.

---

## Autor

**Sergio Antonio Nuñez Lazo**  

---

## Estado del proyecto

**Terminado**
```
