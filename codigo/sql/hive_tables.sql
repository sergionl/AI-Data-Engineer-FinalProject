CREATE DATABASE IF NOT EXISTS yolo_project;
USE yolo_project;

CREATE EXTERNAL TABLE detections (
    detection_id STRING,
    ingestion_date STRING,
    source_type STRING,
    source_id STRING,
    frame_number INT,
    timestamp_sec DOUBLE,
    class_id INT,
    class_name STRING,
    confidence DOUBLE,
    x_min DOUBLE,
    y_min DOUBLE,
    x_max DOUBLE,
    y_max DOUBLE,
    bbox_area DOUBLE,
    bbox_area_ratio DOUBLE,
    center_x DOUBLE,
    center_y DOUBLE,
    position_region STRING,
    dominant_color STRING,
    window_10s INT
)
STORED AS PARQUET
LOCATION '/user/hadoop/yolo_data/';

MSCK REPAIR TABLE detections;