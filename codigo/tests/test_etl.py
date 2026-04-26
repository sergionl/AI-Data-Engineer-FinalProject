import pandas as pd
from unittest.mock import patch

from codigo.batch_etl.sistema_batch_etl import (
    load_checkpoint,
    save_checkpoint,
    clean,
    transform,
    deduplicate,
    load,
)


def test_load_checkpoint_returns_empty_set_if_file_does_not_exist(tmp_path):
    fake_checkpoint = tmp_path / "processed_ids.txt"

    with patch("codigo.batch_etl.sistema_batch_etl.CHECKPOINT_FILE", str(fake_checkpoint)):
        result = load_checkpoint()

    assert result == set()


def test_save_and_load_checkpoint(tmp_path):
    fake_checkpoint = tmp_path / "processed_ids.txt"
    ids = ["id1", "id2", "id3"]

    with patch("codigo.batch_etl.sistema_batch_etl.CHECKPOINT_FILE", str(fake_checkpoint)):
        save_checkpoint(ids)
        loaded = load_checkpoint()

    assert loaded == set(ids)


def test_clean_removes_invalid_rows():
    df = pd.DataFrame({
        "detection_id": ["a", "b", None, "d", "e"],
        "confidence": [0.9, 1.2, 0.5, -0.1, 0.8],
        "x_min": [0, 0, 0, 0, 10],
        "x_max": [10, 10, 10, 10, 5],
        "y_min": [0, 0, 0, 0, 10],
        "y_max": [10, 10, 10, 10, 5],
    })

    result = clean(df)

    assert len(result) == 1
    assert result.iloc[0]["detection_id"] == "a"


def test_transform_casts_types_and_creates_window_10s():
    df = pd.DataFrame({
        "frame_number": [1.0, 20.0, 35.0],
        "class_id": [0.0, 1.0, 2.0],
        "timestamp_sec": [5.0, 12.0, 29.9],
    })

    result = transform(df)

    assert result["frame_number"].dtype == "int64"
    assert result["class_id"].dtype == "int64"
    assert result["window_10s"].tolist() == [0, 1, 2]


def test_deduplicate_removes_processed_and_internal_duplicates():
    df = pd.DataFrame({
        "detection_id": ["a", "b", "b", "c"],
        "value": [1, 2, 3, 4],
    })

    processed_ids = {"a"}

    result = deduplicate(df, processed_ids)

    assert len(result) == 2
    assert result["detection_id"].tolist() == ["b", "c"]


def test_load_writes_image_and_video_parquets(tmp_path):
    df = pd.DataFrame({
        "detection_id": ["1", "2", "3"],
        "source_type": ["image", "video", "video"],
        "source_id": ["img1.jpg", "video 1.mp4", "video 1.mp4"],
        "window_10s": [0, 0, 1],
    })

    with patch("codigo.batch_etl.sistema_batch_etl.OUTPUT_DIR", str(tmp_path)):
        load(df)

    image_file = tmp_path / "images_batch.parquet"
    video_file_1 = tmp_path / "video_1.mp4_window_0.parquet"
    video_file_2 = tmp_path / "video_1.mp4_window_1.parquet"

    assert image_file.exists()
    assert video_file_1.exists()
    assert video_file_2.exists()


def test_load_only_writes_images_file_when_no_videos(tmp_path):
    df = pd.DataFrame({
        "detection_id": ["1"],
        "source_type": ["image"],
        "source_id": ["img1.jpg"],
        "window_10s": [0],
    })

    with patch("codigo.batch_etl.sistema_batch_etl.OUTPUT_DIR", str(tmp_path)):
        load(df)

    image_file = tmp_path / "images_batch.parquet"

    assert image_file.exists()


def test_load_only_writes_video_files_when_no_images(tmp_path):
    df = pd.DataFrame({
        "detection_id": ["2"],
        "source_type": ["video"],
        "source_id": ["video1.mp4"],
        "window_10s": [0],
    })

    with patch("codigo.batch_etl.sistema_batch_etl.OUTPUT_DIR", str(tmp_path)):
        load(df)

    video_file = tmp_path / "video1.mp4_window_0.parquet"

    assert video_file.exists()