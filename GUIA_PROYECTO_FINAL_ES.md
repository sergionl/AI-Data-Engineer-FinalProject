```markdown
# GUÍA DEL PROYECTO FINAL

## 1. Información general

### Título del proyecto
**AI Data Engineer Final Project**

### Autor
**[tu nombre]**

### GitHub
**[tu github]**

### Descripción general
Este proyecto implementa un pipeline de visión computacional y procesamiento batch orientado a ingeniería de datos. El sistema detecta objetos en imágenes y videos utilizando el modelo YOLOv8n, genera un conjunto de detecciones enriquecidas en formato CSV, aplica un proceso ETL batch incremental para limpiar y transformar los datos a formato Parquet, y prepara la información para su posterior carga y análisis en Apache Hive.

El proyecto fue desarrollado con una arquitectura separada en dos componentes principales:

1. un sistema de clasificación de objetos,
2. un sistema batch ETL implementado en Python.

---

## 2. Objetivo del proyecto

El objetivo principal es construir una solución de ingeniería de datos capaz de:

- detectar objetos en imágenes y videos,
- extraer atributos espaciales, temporales y descriptivos de cada detección,
- almacenar resultados intermedios en formato CSV,
- aplicar procesos de limpieza, validación y transformación batch,
- generar salidas analíticas en formato Parquet,
- preparar los datos para ser consultados en Apache Hive,
- evitar registros duplicados en ejecuciones repetidas del pipeline.

---

## 3. Tecnologías utilizadas

- Python 3.10.20
- YOLOv8n
- Pandas
- NumPy
- OpenCV
- PyArrow
- Java
- Hadoop
- Apache Hive
- Makefile
- Pytest
- Pylint
- Black

---

## 4. Requisitos del entorno

Para ejecutar el proyecto se requiere:

- Linux
- Python 3.10.20
- Java
- Hadoop
- Hive

El proyecto utiliza un entorno virtual de Python creado en la carpeta:

```bash
ambiente
```

Los binarios principales definidos en el `Makefile` son:

```bash
./ambiente/venv/bin/python3
./ambiente/venv/bin/pip
```

> Nota: esta configuración refleja el `Makefile` actual del proyecto y debe mantenerse consistente con la estructura real del entorno virtual.

---

## 5. Automatización con Makefile

El proyecto se encuentra automatizado mediante `Makefile`. Los comandos disponibles son los siguientes.

### Mostrar ayuda

```bash
make help
```

Muestra los comandos disponibles definidos en el `Makefile`.

### Crear entorno virtual e instalar dependencias

```bash
make setup
```

Este comando:

1. crea el entorno virtual,
2. actualiza `pip`,
3. instala las dependencias de `requirements.txt`.

### Instalar dependencias

```bash
make install
```

Instala las dependencias listadas en `requirements.txt`.

### Ejecutar pruebas

```bash
make test
```

Ejecuta las pruebas unitarias con `pytest`.

### Ejecutar linting

```bash
make lint
```

Ejecuta análisis estático con `pylint` sobre los módulos del proyecto.

### Formatear el código

```bash
make format
```

Formatea el código con `black`.

### Ejecutar clasificación

```bash
make run-clasificacion
```

Ejecuta el sistema de clasificación de imágenes y videos con YOLO.

### Ejecutar ETL batch

```bash
make run-etl
```

Ejecuta el sistema de procesamiento batch ETL sobre las detecciones generadas.

### Inicializar Hive

```bash
make sql-init
```

Ejecuta el script SQL de Hive:

```bash
codigo/sql/hive_tables.sql
```

### Ejecutar todo el pipeline

```bash
make run-all
```

Ejecuta el pipeline completo en orden:

1. clasificación,
2. ETL batch,
3. carga/definición SQL en Hive.

### Limpiar archivos temporales

```bash
make clean
```

Elimina:

- carpetas `__pycache__`,
- archivos basura `:Zone.Identifier`,
- archivo `derby.log`.

---

## 6. Estructura del repositorio

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

---

## 7. Datos de entrada

Los datos de entrada del proyecto se encuentran en:

```bash
codigo/data
```

El conjunto de datos utilizado incluye:

- al menos 20 imágenes,
- al menos 2 videos.

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

---

## 8. Sistema de clasificación

### Ubicación del código

```bash
codigo/clasificacion/clasificacion.py
```

### Descripción

El sistema de clasificación utiliza el modelo preentrenado **YOLOv8n** para detectar objetos presentes en imágenes y videos. Para cada detección se extrae información estructurada que luego se almacena en un archivo CSV.

### Funcionalidades principales

- lectura automática de imágenes y videos,
- inferencia con YOLOv8n,
- detección de objetos por imagen o frame,
- extracción de atributos por detección,
- consolidación de resultados en CSV.

### Salida

La salida del módulo de clasificación es:

```bash
codigo/csv/clasificacion/detections.csv
```

---

## 9. Esquema de datos generado

Cada detección contiene los siguientes campos:

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

Estos atributos permiten representar información relevante sobre ubicación, tamaño, proporción, color y tiempo de cada objeto detectado.

---

## 10. Sistema ETL batch

### Ubicación del código

```bash
codigo/batch_etl/sistema_batch_etl.py
```

### Entrada

El ETL consume el archivo:

```bash
codigo/csv/clasificacion/detections.csv
```

### Transformaciones realizadas

El proceso ETL aplica las siguientes reglas:

- eliminación de valores nulos,
- eliminación de confidencias fuera del rango válido `[0,1]`,
- eliminación de bounding boxes corruptos:
  - `x_max <= x_min`
  - `y_max <= y_min`
- normalización de algunos datos,
- creación de la columna `window_10s`,
- control de duplicados utilizando `detection_id`.

### Salida

La salida del ETL se almacena en formato Parquet en:

```bash
codigo/csv/batch_etl
```

---

## 11. Estrategia de incrementalidad y deduplicación

El sistema batch utiliza un mecanismo de checkpoint basado en el archivo:

```bash
codigo/batch_etl/processed_ids.txt
```

### Funcionamiento del checkpoint

- si el archivo existe, se leen todos los `detection_id` previamente procesados,
- estos identificadores se almacenan en un conjunto para validación eficiente,
- si el archivo no existe, se trabaja con un conjunto vacío,
- solo se procesan detecciones nuevas,
- si el ETL se ejecuta dos veces sobre el mismo conjunto de datos, no se producen cambios adicionales.

Esta lógica permite mantener idempotencia y evitar duplicidad de registros.

---

## 12. Integración con Hive

### Archivos SQL

Los archivos SQL del proyecto se encuentran en:

```bash
codigo/sql/hive_tables.sql
codigo/sql/consultas_hive.sql
```

### Propósito

La tabla principal `detections` fue diseñada para almacenar la salida procesada del pipeline y facilitar análisis en Apache Hive sobre:

- clases detectadas,
- tipos de fuente,
- regiones dentro del frame,
- ventanas temporales de 10 segundos,
- colores dominantes,
- métricas espaciales derivadas.

La inicialización de la tabla se realiza con:

```bash
make sql-init
```

que ejecuta internamente:

```bash
hive -f codigo/sql/hive_tables.sql
```

---

## 13. Flujo del pipeline

El flujo general del proyecto es el siguiente:

1. se colocan imágenes y videos en `codigo/data`,
2. el sistema de clasificación procesa los archivos y genera `detections.csv`,
3. el sistema ETL limpia, valida y transforma la información,
4. la salida se exporta a formato Parquet,
5. los scripts SQL permiten definir tablas y ejecutar consultas en Hive.

La ejecución completa puede realizarse con:

```bash
make run-all
```

---

## 14. Pruebas

El proyecto incluye pruebas en la carpeta:

```bash
tests
```

Archivos de prueba incluidos:

- `test_clasificacion.py`
- `test_etl.py`
- `test_files.py`

Estas pruebas permiten validar componentes del pipeline y verificar que la estructura del proyecto sea consistente.

Para ejecutar las pruebas:

```bash
make test
```

---

## 15. Instalación de dependencias

Para preparar el entorno de forma automatizada:

```bash
make setup
```

Para instalar dependencias manualmente:

```bash
pip install -r requirements.txt
```

O usando el `Makefile`:

```bash
make install
```

---

## 16. Consultas analíticas

Las consultas analíticas del proyecto se encuentran en:

```bash
codigo/sql/consultas_hive.sql
```

Estas consultas permiten realizar análisis como:

- conteo de objetos por clase,
- distribución por tipo de fuente,
- análisis por ventanas de 10 segundos,
- detección por región dentro del frame,
- análisis de color dominante por clase.

---

## 17. Consideraciones finales

El proyecto fue desarrollado bajo un enfoque de ingeniería de datos aplicada a visión computacional, integrando detección de objetos, almacenamiento intermedio estructurado, procesamiento batch, control incremental y preparación de datos para análisis sobre Hive.

La solución cumple con la separación de responsabilidades entre inferencia y transformación de datos, permitiendo trazabilidad, reproducibilidad y automatización del flujo completo.

---
