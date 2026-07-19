from pathlib import Path


class Config:
    """
    Global project configuration.
    """

    # ------------------------------------------------------------------
    # Project Paths
    # ------------------------------------------------------------------

    ROOT_DIR = Path(__file__).resolve().parent.parent

    DATA_DIR = ROOT_DIR / "data"

    RAW_DATA_DIR = DATA_DIR / "raw"

    PROCESSED_DATA_DIR = DATA_DIR / "processed"

    MODEL_DIR = ROOT_DIR / "models"

    NOTEBOOK_DIR = ROOT_DIR / "notebooks"

    APP_DIR = ROOT_DIR / "app"

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    DATASET_NAME = "heart_disease_data.csv"

    DATASET_PATH = RAW_DATA_DIR / DATASET_NAME

    TARGET_COLUMN = "target"

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    MODEL_NAME = "best.pkl"

    MODEL_PATH = MODEL_DIR / MODEL_NAME

    # ------------------------------------------------------------------
    # Experiment
    # ------------------------------------------------------------------

    RANDOM_STATE = 42

    TEST_SIZE = 0.20

    CV = 5

    SCORING = "f1"