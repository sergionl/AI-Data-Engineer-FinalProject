# =========================
# IMPORTS
# =========================
import cv2
import pandas as pd
from ultralytics import YOLO
import os
import numpy as np
from datetime import datetime
import uuid
from pathlib import Path

# =========================
# CONFIG
# =========================

script_dir = Path(__file__).parent
images_path = script_dir.parent / "data" / "images"
IMAGES_PATH = str(images_path)

videos_path = script_dir.parent / "data" / "videos"
VIDEOS_PATH = str(videos_path)

csv_path = script_dir.parent / "csv" / "clasificacion" / "detections.csv"
CSV_PATH = str(csv_path)

model = None


def get_model():
    """Carga el modelo YOLO solo cuando se necesita."""
    global model
    if model is None:
        model = YOLO("yolov8n.pt")
    return model


# =========================
# COLOR
# =========================

def get_color_name(r, g, b):
    """Determina un nombre de color simple a partir de valores RGB."""
    if r > 150 and g < 100 and b < 100:
        return "red"
    if g > 150 and r < 100:
        return "green"
    if b > 150 and r < 100:
        return "blue"
    if r > 200 and g > 200 and b > 200:
        return "white"
    if r < 50 and g < 50 and b < 50:
        return "black"
    return "other"


# =========================
# EXTRACTOR
# =========================

def extract_detections(results, frame, source_type, source_id, frame_number, model_obj, fps=None):
    """Extrae detecciones desde la salida del modelo y construye registros tabulares."""
    records = []

    if not results:
        return records

    result = results[0]

    if result.boxes is None:
        return records

    frame_height, frame_width = result.orig_shape[:2]

    for box in result.boxes:
        ingestion_date = datetime.now().isoformat()
        detection_id = str(uuid.uuid4())

        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

        width = x_max - x_min
        height = y_max - y_min
        area_pixels = width * height

        bbox_area_ratio = area_pixels / (frame_height * frame_width)

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        center_x_norm = center_x / frame_width
        center_y_norm = center_y / frame_height

        vertical = "top" if center_y_norm < 0.33 else "middle" if center_y_norm < 0.66 else "bottom"
        horizontal = "left" if center_x_norm < 0.33 else "center" if center_x_norm < 0.66 else "right"
        position_region = f"{vertical}-{horizontal}"

        x1, y1, x2, y2 = map(int, [x_min, y_min, x_max, y_max])
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            avg_color = [0, 0, 0]
        else:
            avg_color = np.mean(roi, axis=(0, 1))

        b, g, r_color = avg_color
        dominant_color_name = get_color_name(r_color, g, b)

        timestamp_sec = frame_number / fps if (source_type == "video" and fps) else 0

        record = {
            "detection_id": detection_id,
            "ingestion_date": ingestion_date,
            "source_type": source_type,
            "source_id": source_id,
            "frame_number": frame_number,
            "class_id": int(box.cls),
            "class_name": model_obj.names[int(box.cls)],
            "confidence": float(box.conf),
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "width": width,
            "height": height,
            "area_pixels": area_pixels,
            "frame_height": frame_height,
            "frame_width": frame_width,
            "bbox_area_ratio": bbox_area_ratio,
            "center_x": center_x,
            "center_y": center_y,
            "center_x_norm": center_x_norm,
            "center_y_norm": center_y_norm,
            "position_region": position_region,
            "dominant_color_name": dominant_color_name,
            "dom_r": int(r_color),
            "dom_g": int(g),
            "dom_b": int(b),
            "timestamp_sec": timestamp_sec,
        }

        records.append(record)

    return records


# =========================
# IMÁGENES
# =========================

def process_images(path):
    """Procesa todas las imágenes válidas de un directorio."""
    all_records = []
    model_obj = get_model()

    for file in os.listdir(path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(path, file)

            image = cv2.imread(image_path)
            if image is None:
                continue

            results = model_obj(image)

            records = extract_detections(
                results, image, "image", file, 0, model_obj
            )

            all_records.extend(records)

    return all_records


# =========================
# VIDEOS
# =========================

def process_videos(path):
    """Procesa todos los videos válidos de un directorio frame por frame."""
    all_records = []
    model_obj = get_model()

    for file in os.listdir(path):
        if file.lower().endswith((".mp4", ".avi", ".mov")):
            video_path = os.path.join(path, file)

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            frame_number = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model_obj(frame)

                records = extract_detections(
                    results, frame, "video", file, frame_number, model_obj, fps
                )

                all_records.extend(records)
                frame_number += 1

            cap.release()

    return all_records


# =========================
# MAIN
# =========================

def main():
    """Ejecuta el pipeline completo de clasificación y exporta a CSV."""
    data = []
    data.extend(process_images(IMAGES_PATH))
    data.extend(process_videos(VIDEOS_PATH))

    df = pd.DataFrame(data)
    Path(CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)

    print("CSV generado con éxito")


if __name__ == "__main__":
    main()