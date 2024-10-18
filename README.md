# Reco Algo

Try to implement some Recommendation Algorithm with PyTorch to learn both.

# Prerequisite
- Poetry 1.8.3
- Miniconda or alternatives that can create new Python environment with a specified Python version

# Set up
- Create a new `.env` file based on `.env.example` and populate the variables there
- Create a new Python 3.11.9 environment: `conda create --prefix .venv python=3.11.9`
- Make sure Poetry use the new Python 3.11.9 environment: `poetry env use .venv/bin/python`
- Install Python dependencies with Poetry: `poetry install`

# Run
## Compare different reco algos
- Start MLflow locally: `make mlflow-up`
- Start the Jupyterlab notebook: `make notebook-up`
- Run the data prep notebooks in this order: [000](./notebooks/000-prep-data.ipynb), 001, 002.
- Then you can run the notebooks from 010 to see how the algorithms fit with the prep data.
- Experiment evaluation should be available via MLflow web UI at localhost:5003 if you have run `make mlflow-up`.

## Try Item2Vec modeling to learn item embeddings
- Run notebooks 020 to 025
- 