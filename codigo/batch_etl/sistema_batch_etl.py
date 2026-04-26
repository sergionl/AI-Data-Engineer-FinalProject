# =========================
# IMPORTS
# =========================
import pandas as pd
import os
from pathlib import Path

# =========================
# CONFIG
# =========================

script_dir = Path(__file__).parent

input_csv = script_dir.parent / "csv" / "clasificacion" / "detections.csv"
INPUT_CSV = str(input_csv)

output_dir = script_dir.parent / "csv" / "batch_etl"
OUTPUT_DIR = str(output_dir)

checkpoint_file = script_dir / "processed_ids.txt"
CHECKPOINT_FILE = str(checkpoint_file)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# CHECKPOINT
# =========================

def load_checkpoint():
    """Carga el checkpoint de IDs ya procesados."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as file:
        return set(line.strip() for line in file)


def save_checkpoint(new_ids):
    """Guarda nuevos IDs procesados en el checkpoint."""
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as file:
        for item_id in new_ids:
            file.write(f"{item_id}\n")


# =========================
# EXTRACT
# =========================

def extract():
    """Extrae los datos desde el CSV de detecciones."""
    print("Extracting data...")
    df = pd.read_csv(INPUT_CSV)
    print(f"   → {len(df)} registros cargados")
    return df


# =========================
# CLEAN
# =========================

def clean(df):
    """Limpia registros nulos, confidencias inválidas y bounding boxes corruptos."""
    print("Cleaning data...")

    initial = len(df)

    df = df.dropna(subset=["detection_id", "confidence"])
    df = df[(df["confidence"] >= 0) & (df["confidence"] <= 1)]
    df = df[(df["x_max"] > df["x_min"]) & (df["y_max"] > df["y_min"])]

    print(f"   → {len(df)} registros después de limpieza ({initial - len(df)} eliminados)")
    return df


# =========================
# TRANSFORM
# =========================

def transform(df):
    """Transforma tipos de datos y crea ventanas de 10 segundos."""
    print("Transforming data...")

    df = df.copy()
    df["frame_number"] = df["frame_number"].astype(int)
    df["class_id"] = df["class_id"].astype(int)
    df["window_10s"] = (df["timestamp_sec"] // 10).astype(int)

    return df


# =========================
# DEDUPLICATION
# =========================

def deduplicate(df, processed_ids):
    """Elimina registros ya procesados y duplicados internos por detection_id."""
    print("Removing duplicates...")

    before = len(df)

    df = df[~df["detection_id"].isin(processed_ids)]
    df = df.drop_duplicates(subset=["detection_id"])

    print(f"   → {len(df)} registros después de deduplicación ({before - len(df)} eliminados)")
    return df


# =========================
# LOAD
# =========================

def load(df):
    """Carga la data procesada a archivos parquet separados para imágenes y videos."""
    print("Loading data...")

    images_df = df[df["source_type"] == "image"]
    videos_df = df[df["source_type"] == "video"]

    if not images_df.empty:
        path = os.path.join(OUTPUT_DIR, "images_batch.parquet")
        images_df.to_parquet(path, index=False)
        print(f"Imágenes guardadas en {path}")

    if not videos_df.empty:
        grouped = videos_df.groupby(["source_id", "window_10s"])

        for source_id, window in grouped.groups:
            group = grouped.get_group((source_id, window))
            safe_source_id = str(source_id).replace(" ", "_")
            filename = f"{safe_source_id}_window_{window}.parquet"
            path = os.path.join(OUTPUT_DIR, filename)
            group.to_parquet(path, index=False)

        print("Videos guardados por ventanas de 10s")


# =========================
# MAIN
# =========================

def main():
    """Ejecuta el pipeline ETL batch completo."""
    print("Iniciando ETL...\n")

    processed_ids = load_checkpoint()

    df = extract()
    df = clean(df)
    df = transform(df)
    df = deduplicate(df, processed_ids)

    if df.empty:
        print("No hay nuevos datos para procesar")
        return

    load(df)
    save_checkpoint(df["detection_id"])

    print("\nETL completado correctamente")


if __name__ == "__main__":
    main()