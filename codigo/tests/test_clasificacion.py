import numpy as np
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from codigo.clasificacion.clasificacion import (
    get_color_name,
    extract_detections,
    process_images,
    process_videos,
)


def test_get_color_name_red():
    assert get_color_name(200, 50, 50) == "red"


def test_get_color_name_green():
    assert get_color_name(50, 200, 50) == "green"


def test_get_color_name_blue():
    assert get_color_name(50, 50, 200) == "blue"


def test_get_color_name_white():
    assert get_color_name(255, 255, 255) == "white"


def test_get_color_name_black():
    assert get_color_name(0, 0, 0) == "black"


def test_get_color_name_other():
    assert get_color_name(120, 120, 120) == "other"


def test_extract_detections_returns_empty_when_no_results():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = extract_detections(
        [],
        frame,
        "image",
        "img1.jpg",
        0,
        model_obj=SimpleNamespace(names={0: "person"})
    )
    assert result == []


def test_extract_detections_returns_empty_when_boxes_is_none():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    fake_result = SimpleNamespace(
        boxes=None,
        orig_shape=(100, 100, 3)
    )

    result = extract_detections(
        [fake_result],
        frame,
        "image",
        "img1.jpg",
        0,
        model_obj=SimpleNamespace(names={0: "person"})
    )

    assert result == []


def test_extract_detections_builds_expected_record():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:50, 20:60] = [0, 0, 255]

    fake_box = SimpleNamespace(
        xyxy=np.array([[20, 10, 60, 50]], dtype=float),
        cls=0,
        conf=0.95
    )

    fake_result = SimpleNamespace(
        boxes=[fake_box],
        orig_shape=(100, 100, 3)
    )

    fake_model = SimpleNamespace(names={0: "person"})

    records = extract_detections(
        [fake_result],
        frame,
        "image",
        "test.jpg",
        0,
        fake_model
    )

    assert len(records) == 1

    record = records[0]

    assert record["source_type"] == "image"
    assert record["source_id"] == "test.jpg"
    assert record["frame_number"] == 0
    assert record["class_id"] == 0
    assert record["class_name"] == "person"
    assert record["confidence"] == 0.95
    assert record["x_min"] == 20
    assert record["y_min"] == 10
    assert record["x_max"] == 60
    assert record["y_max"] == 50
    assert record["width"] == 40
    assert record["height"] == 40
    assert record["area_pixels"] == 1600
    assert 0 <= record["center_x_norm"] <= 1
    assert 0 <= record["center_y_norm"] <= 1
    assert record["dominant_color_name"] == "red"
    assert record["timestamp_sec"] == 0
    assert "detection_id" in record
    assert "ingestion_date" in record


def test_extract_detections_computes_video_timestamp():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    fake_box = SimpleNamespace(
        xyxy=np.array([[10, 10, 30, 30]], dtype=float),
        cls=0,
        conf=0.80
    )

    fake_result = SimpleNamespace(
        boxes=[fake_box],
        orig_shape=(100, 100, 3)
    )

    fake_model = SimpleNamespace(names={0: "car"})

    records = extract_detections(
        [fake_result],
        frame,
        "video",
        "video1.mp4",
        frame_number=30,
        model_obj=fake_model,
        fps=10
    )

    assert len(records) == 1
    assert records[0]["timestamp_sec"] == 3.0


def test_extract_detections_handles_empty_roi():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    fake_box = SimpleNamespace(
        xyxy=np.array([[90, 90, 90, 90]], dtype=float),
        cls=0,
        conf=0.70
    )

    fake_result = SimpleNamespace(
        boxes=[fake_box],
        orig_shape=(100, 100, 3)
    )

    fake_model = SimpleNamespace(names={0: "dog"})

    records = extract_detections(
        [fake_result],
        frame,
        "image",
        "img.jpg",
        0,
        fake_model
    )

    assert len(records) == 1
    assert records[0]["dominant_color_name"] == "black"
    assert records[0]["dom_r"] == 0
    assert records[0]["dom_g"] == 0
    assert records[0]["dom_b"] == 0


@patch("codigo.clasificacion.clasificacion.get_model")
@patch("codigo.clasificacion.clasificacion.extract_detections")
@patch("codigo.clasificacion.clasificacion.cv2.imread")
@patch("codigo.clasificacion.clasificacion.os.listdir")
def test_process_images_only_processes_image_files(mock_listdir, mock_imread, mock_extract, mock_get_model):
    mock_listdir.return_value = ["a.jpg", "b.png", "c.txt"]
    mock_imread.return_value = np.zeros((50, 50, 3), dtype=np.uint8)

    fake_model = MagicMock()
    fake_model.return_value = ["fake_results"]
    mock_get_model.return_value = fake_model

    mock_extract.return_value = [{"detection_id": "1"}]

    result = process_images("fake_path")

    assert len(result) == 2
    assert mock_imread.call_count == 2
    assert mock_extract.call_count == 2


@patch("codigo.clasificacion.clasificacion.os.listdir")
@patch("codigo.clasificacion.clasificacion.cv2.imread")
@patch("codigo.clasificacion.clasificacion.get_model")
def test_process_images_skips_invalid_images(mock_get_model, mock_imread, mock_listdir):
    mock_listdir.return_value = ["bad.jpg"]
    mock_imread.return_value = None
    mock_get_model.return_value = MagicMock()

    result = process_images("fake_path")

    assert result == []


@patch("codigo.clasificacion.clasificacion.get_model")
@patch("codigo.clasificacion.clasificacion.extract_detections")
@patch("codigo.clasificacion.clasificacion.os.listdir")
@patch("codigo.clasificacion.clasificacion.cv2.VideoCapture")
def test_process_videos_processes_video_frames(mock_videocapture, mock_listdir, mock_extract, mock_get_model):
    mock_listdir.return_value = ["video1.mp4", "notes.txt"]

    fake_cap = MagicMock()
    fake_cap.get.return_value = 10
    fake_cap.read.side_effect = [
        (True, np.zeros((50, 50, 3), dtype=np.uint8)),
        (True, np.zeros((50, 50, 3), dtype=np.uint8)),
        (False, None),
    ]
    mock_videocapture.return_value = fake_cap

    fake_model = MagicMock()
    fake_model.return_value = ["fake_results"]
    mock_get_model.return_value = fake_model

    mock_extract.return_value = [{"detection_id": "1"}]

    result = process_videos("fake_path")

    assert len(result) == 2
    assert mock_extract.call_count == 2
    fake_cap.release.assert_called_once()