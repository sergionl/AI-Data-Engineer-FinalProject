CREATE DATABASE IF NOT EXISTS yolo_project;
USE yolo_project;

CREATE EXTERNAL TABLE IF NOT EXISTS detections (
    source_type STRING,
    source_id STRING,
    frame_number INT,
    class_id INT,
    class_name STRING,
    confidence DOUBLE,

    x_min INT,
    y_min INT,
    x_max INT,
    y_max INT,

    width INT,
    height INT,
    area_pixels INT,

    frame_width INT,
    frame_height INT,
    bbox_area_ratio DOUBLE,

    center_x DOUBLE,
    center_y DOUBLE,
    center_x_norm DOUBLE,
    center_y_norm DOUBLE,

    position_region STRING,

    dominant_color_name STRING,
    dom_r INT,
    dom_g INT,
    dom_b INT,

    timestamp_sec DOUBLE,
    ingestion_date STRING,
    detection_id STRING
)
STORED AS PARQUET
LOCATION '/user/hadoop/yolo_data/';

MSCK REPAIR TABLE detections;