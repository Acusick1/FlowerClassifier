import os
import pathlib

# Paths
parent_path = pathlib.Path(os.path.dirname(__file__))
DATASET_DIR = parent_path.joinpath("../datasets").resolve()
MODEL_DIR = parent_path.joinpath("../saved_models").resolve()
pathlib.Path.mkdir(pathlib.Path(MODEL_DIR), exist_ok=True)

SEED = 0
# Model
BATCH_SIZE = 32
IMG_SIZE = (256, 256, 3)

MODELS = [f.name for f in MODEL_DIR.iterdir() if f.is_dir()]
DATASETS = [f.name for f in DATASET_DIR.iterdir() if f.is_dir()]
