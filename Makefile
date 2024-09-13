.PHONY:
.ONESHELL:

include .env
export

mlflow-up:
	poetry run mlflow server --host 0.0.0.0 --port 5003 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
