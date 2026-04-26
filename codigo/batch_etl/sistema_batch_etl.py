"""
Sistema Batch ETL para procesamiento de detecciones.

Este módulo implementa la segunda parte del proyecto: el sistema batch ETL.
Su objetivo es tomar el archivo CSV generado por el sistema de clasificación,
aplicar un proceso de extracción, limpieza, transformación y carga, y
persistir los resultados en archivos Parquet listos para análisis posterior.

Además, el script mantiene un mecanismo de checkpoint basado en IDs ya
procesados para evitar reprocesar registros en ejecuciones futuras. El flujo
separa los datos provenientes de imágenes y videos:
- Las detecciones de imágenes se almacenan en un único archivo Parquet.
- Las detecciones de videos se almacenan en múltiples archivos Parquet
  agrupados por video y por ventanas de 10 segundos. [github.com](https://github.com/sergionl/AI-Data-Engineer-FinalProject/blob/main/codigo/batch_etl/sistema_batch_etl.py)

Flujo general del pipeline:
1. Cargar el checkpoint de detecciones previamente procesadas.
2. Extraer los datos desde el CSV de clasificación.
3. Limpiar registros inválidos o corruptos.
4. Transformar tipos de datos y generar ventanas temporales de 10 segundos.
5. Eliminar duplicados y registros ya procesados.
6. Cargar la salida en formato Parquet.
7. Actualizar el checkpoint con los nuevos detection_id procesados. [github.com](https://github.com/sergionl/AI-Data-Engineer-FinalProject/blob/main/codigo/batch_etl/sistema_batch_etl.py)
"""

# =========================
# IMPORTS
# =========================
import os
from pathlib import Path

import pandas as pd

# =========================
# CONFIGURACIÓN DE RUTAS
# =========================
# Se definen rutas relativas al archivo actual para que el script pueda
# ejecutarse en distintos entornos sin depender de rutas absolutas.
script_dir = Path(__file__).parent

# Archivo CSV generado por la etapa de clasificación.
input_csv = script_dir.parent / "csv" / "clasificacion" / "detections.csv"
INPUT_CSV = str(input_csv)

# Directorio donde se guardarán los archivos parquet de salida.
output_dir = script_dir.parent / "csv" / "batch_etl"
OUTPUT_DIR = str(output_dir)

# Archivo de checkpoint que almacena los detection_id ya procesados.
checkpoint_file = script_dir / "processed_ids.txt"
CHECKPOINT_FILE = str(checkpoint_file)

# Se asegura que exista el directorio de salida antes de comenzar.
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# CHECKPOINT
# =========================
def load_checkpoint():
    """
    Carga el checkpoint de IDs ya procesados.

    El checkpoint permite que el pipeline batch sea incremental. Si el archivo
    existe, se leen todos los detection_id guardados y se devuelven en un set
    para consulta eficiente. Si no existe, se retorna un conjunto vacío.

    Returns:
        set[str]: Conjunto de IDs de detecciones ya procesadas.
    """
    if not os.path.exists(CHECKPOINT_FILE):
        return set()

    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as file:
        return set(line.strip() for line in file)


def save_checkpoint(new_ids):
    """
    Guarda nuevos IDs procesados en el checkpoint.

    Cada detection_id se agrega en una nueva línea del archivo para que pueda
    ser utilizado en futuras ejecuciones del pipeline y así evitar
    reprocesamiento.

    Args:
        new_ids (iterable): Colección de IDs nuevos procesados.

    Returns:
        None
    """
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as file:
        for item_id in new_ids:
            file.write(f"{item_id}\n")


# =========================
# EXTRACT
# =========================
def extract():
    """
    Extrae los datos desde el CSV de detecciones.

    Esta etapa corresponde a la fase de extracción del pipeline ETL. Carga
    en memoria el archivo CSV producido por el sistema de clasificación y
    devuelve su contenido en un DataFrame de pandas.

    Returns:
        pandas.DataFrame: DataFrame con todas las detecciones extraídas.
    """
    print("Extracting data...")

    df = pd.read_csv(INPUT_CSV)

    print(f" → {len(df)} registros cargados")
    return df


# =========================
# CLEAN
# =========================
def clean(df):
    """
    Limpia registros inválidos o inconsistentes.

    Se eliminan:
    - filas con valores nulos en campos clave,
    - confidencias fuera del rango válido [0, 1],
    - bounding boxes corruptos donde x_max <= x_min o y_max <= y_min.

    Args:
        df (pandas.DataFrame): DataFrame original extraído.

    Returns:
        pandas.DataFrame: DataFrame depurado.
    """
    print("Cleaning data...")

    initial = len(df)

    # Eliminación de registros sin identificador o sin confidence,
    # ya que son campos esenciales para trazabilidad y calidad.
    df = df.dropna(subset=["detection_id", "confidence"])

    # Validación del rango lógico de confianza del modelo.
    df = df[(df["confidence"] >= 0) & (df["confidence"] <= 1)]

    # Validación geométrica del bounding box.
    df = df[(df["x_max"] > df["x_min"]) & (df["y_max"] > df["y_min"])]

    print(f" → {len(df)} registros después de limpieza ({initial - len(df)} eliminados)")
    return df


# =========================
# TRANSFORM
# =========================
def transform(df):
    """
    Transforma tipos de datos y genera atributos derivados.

    En esta fase se normalizan algunos tipos de datos relevantes y se crea
    la columna `window_10s`, que agrupa los registros de video en ventanas
    temporales de 10 segundos para la salida batch.

    Args:
        df (pandas.DataFrame): DataFrame limpio.

    Returns:
        pandas.DataFrame: DataFrame transformado con atributos derivados.
    """
    print("Transforming data...")

    df = df.copy()

    # Conversión de tipos para asegurar consistencia semántica.
    df["frame_number"] = df["frame_number"].astype(int)
    df["class_id"] = df["class_id"].astype(int)

    # Cálculo de ventana temporal discreta basada en timestamp_sec.
    # Ejemplo:
    #   0.0 a 9.999  -> ventana 0
    #   10.0 a 19.999 -> ventana 1
    df["window_10s"] = (df["timestamp_sec"] // 10).astype(int)

    return df


# =========================
# DEDUPLICATION
# =========================
def deduplicate(df, processed_ids):
    """
    Elimina registros duplicados o ya procesados anteriormente.

    Primero se excluyen todos los registros cuyo detection_id ya está en el
    checkpoint. Luego se eliminan duplicados internos dentro del mismo lote
    usando detection_id como clave única.

    Args:
        df (pandas.DataFrame): DataFrame transformado.
        processed_ids (set[str]): IDs previamente procesados.

    Returns:
        pandas.DataFrame: DataFrame sin registros repetidos.
    """
    print("Removing duplicates...")

    before = len(df)

    # Se excluyen los registros que ya fueron procesados en corridas previas.
    df = df[~df["detection_id"].isin(processed_ids)]

    # Se eliminan duplicados dentro del lote actual.
    df = df.drop_duplicates(subset=["detection_id"])

    print(f" → {len(df)} registros después de deduplicación ({before - len(df)} eliminados)")
    return df


# =========================
# LOAD
# =========================
def load(df):
    """
    Carga la data procesada en archivos Parquet.

    Estrategia de carga:
    - Las detecciones provenientes de imágenes se guardan en un único archivo
      llamado `images_batch.parquet`.
    - Las detecciones provenientes de videos se agrupan por `source_id` y
      `window_10s`, generando un archivo Parquet por cada combinación.

    Args:
        df (pandas.DataFrame): DataFrame final listo para persistencia.

    Returns:
        None
    """
    print("Loading data...")

    # Separación por tipo de fuente para aplicar reglas distintas de salida.
    images_df = df[df["source_type"] == "image"]
    videos_df = df[df["source_type"] == "video"]

    # =========================
    # CARGA DE IMÁGENES
    # =========================
    if not images_df.empty:
        path = os.path.join(OUTPUT_DIR, "images_batch.parquet")
        images_df.to_parquet(path, index=False)
        print(f"Imágenes guardadas en {path}")

    # =========================
    # CARGA DE VIDEOS
    # =========================
    # En videos, la lógica de negocio requiere particionar por video y por
    # ventana de 10 segundos para facilitar consultas posteriores.
    if not videos_df.empty:
        grouped = videos_df.groupby(["source_id", "window_10s"])

        for source_id, window in grouped.groups:
            group = grouped.get_group((source_id, window))

            # Se reemplazan espacios para generar nombres de archivo seguros.
            safe_source_id = str(source_id).replace(" ", "_")
            filename = f"{safe_source_id}_window_{window}.parquet"
            path = os.path.join(OUTPUT_DIR, filename)

            group.to_parquet(path, index=False)

        print("Videos guardados por ventanas de 10s")


# =========================
# MAIN
# =========================
def main():
    """
    Ejecuta el pipeline ETL batch completo.

    Flujo principal:
    1. Carga checkpoint de IDs procesados.
    2. Extrae los datos desde el CSV.
    3. Limpia registros inválidos.
    4. Transforma tipos y genera ventanas temporales.
    5. Deduplica respecto a histórico y al lote actual.
    6. Si hay datos nuevos, los carga a Parquet.
    7. Guarda en checkpoint los nuevos IDs procesados.

    Returns:
        None
    """
    print("Iniciando ETL...\n")

    # Carga de IDs históricos ya procesados.
    processed_ids = load_checkpoint()

    # Fases del pipeline ETL.
    df = extract()
    df = clean(df)
    df = transform(df)
    df = deduplicate(df, processed_ids)

    # Si después de la depuración no quedan registros nuevos,
    # se termina la ejecución de forma controlada.
    if df.empty:
        print("No hay nuevos datos para procesar")
        return

    # Persistencia final de la salida batch.
    load(df)

    # Actualización del checkpoint para futuras ejecuciones.
    save_checkpoint(df["detection_id"])

    print("\nETL completado correctamente")


# Punto de entrada del script.
if __name__ == "__main__":
    main()