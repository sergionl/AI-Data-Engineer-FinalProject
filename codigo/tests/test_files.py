from pathlib import Path


def test_clasificacion_file_exists():
    assert Path("codigo/clasificacion/clasificacion.py").exists()


def test_etl_file_exists():
    assert Path("codigo/batch_etl/sistema_batch_etl.py").exists()


def test_requirements_exists():
    assert Path("requirements.txt").exists()


def test_makefile_exists():
    assert Path("Makefile").exists()


def test_hive_sql_exists():
    assert Path("codigo/sql/hive_tables.sql").exists()