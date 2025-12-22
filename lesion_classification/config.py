from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
import torch


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_DIR: Path = PROJECT_ROOT / "data"
    TRAIN_DIR: Path = DATA_DIR / "train"
    UNLABELED_DIR: Path = DATA_DIR / "unlabeled"
    VAL_DIR: Path = DATA_DIR / "val"

    # Training Hyperparameters
    BATCH_SIZE: int = 32
    EPOCHS: int = 20
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    EMA_DECAY: float = 0.99

    # Model Settings
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    PRE_TRAINED: bool = True
    DROPOUT: float = 0.5
    NUM_CLASSES: int = 1

    # Loss Settings
    FOCAL_GAMMA: float = 2.0
    FOCAL_ALPHA: float = 0.6

    # Data settings
    TRAIN_SPLIT_SIZE: int = 270
    UNLABELED_SPLIT_SIZE: int = 540
    SPLIT_SEED: int = 42
    IMAGE_RESIZE: int = 256
    IMAGE_SIZE: int = 224
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = torch.cuda.is_available()


settings = Settings()


def ensure_dirs() -> None:
    """Ensure that the necessary directories exist."""
    settings.TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    settings.UNLABELED_DIR.mkdir(parents=True, exist_ok=True)
    settings.VAL_DIR.mkdir(parents=True, exist_ok=True)

    (settings.TRAIN_DIR / "benign").mkdir(parents=True, exist_ok=True)
    (settings.TRAIN_DIR / "malignant").mkdir(parents=True, exist_ok=True)
    (settings.VAL_DIR / "benign").mkdir(parents=True, exist_ok=True)
    (settings.VAL_DIR / "malignant").mkdir(parents=True, exist_ok=True)
