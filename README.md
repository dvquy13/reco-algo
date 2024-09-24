Try to implement some Recommendation Algorithm with PyTorch to learn both.

# Set up

## Prerequisite
- Poetry 1.8.3

## Install environment
```shell
conda create --prefix .venv python=3.11.9
poetry env use .venv/bin/python
poetry install
cp .env.example .env
```

# How to start?

```shell
make mlflow-up
make notebook-up
```

Run the data prep notebooks in this order: [000](./notebooks/000-prep-data.ipynb), 001, 002.

Then you can run the notebooks from 010 to see how the algorithms fit with the prep data.

Experiment evaluation should be available via MLflow web UI at localhost:5003 if you have run `make mlflow-up`.
