# Variables
PYTHON = ./ambiente/bin/python3
PIP = ./ambiente/bin/pip

.PHONY: setup install run-clasificacion run-etl sql-init run-all clean help

help: ## Muestra los comandos disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Crea el entorno virtual e instala dependencias
	python3 -m venv ambiente
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install: ## Instala las dependencias del requirements.txt
	$(PIP) install -r requirements.txt

run-clasificacion: ## Paso 1: Clasificación de imágenes/videos (YOLO)
	$(PYTHON) codigo/clasificacion/clasificacion.py

run-etl: ## Paso 2: Procesamiento Batch ETL de los resultados
	$(PYTHON) codigo/batch-etl/sistema_batch_etl.py

sql-init: ## Paso 3: Carga de datos en Hive
	hive -f sql/hive_tables.sql

run-all: run-clasificacion run-etl sql-init ## Ejecuta el pipeline completo en orden: Clasificación > ETL > Hive
	@echo "-------------------------------------------"
	@echo "Pipeline finalizado con éxito en el orden requerido."
	@echo "-------------------------------------------"

clean: ## Limpia archivos temporales, logs y basura de Windows (:Zone.Identifier)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*:Zone.Identifier" -type f -delete
	rm -f derby.log
	@echo "Limpieza completada."
