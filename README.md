Try to implement some Recommendation Algorithm with PyTorch to learn both.

# Set up

## Prerequisite
- Poetry 1.8.3

```
conda create --prefix .venv python=3.11.9
poetry env use .venv
poetry install
cp .env.example .env
make mlflow-up
```