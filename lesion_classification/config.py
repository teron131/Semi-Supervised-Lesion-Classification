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

    # Model Settings
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    PRE_TRAINED: bool = True
    NUM_CLASSES: int = 1

    # Loss Settings
    FOCAL_GAMMA: float = 2.0
    FOCAL_ALPHA: float = 0.6
    SUPERVISED_LOSS: str = "bce"  # "focal" or "bce"
    POS_WEIGHT: float = 3.0
    AUTO_POS_WEIGHT: bool = True
    POS_WEIGHT_MAX: float = 3.0

    # Data settings
    TRAIN_SPLIT_SIZE: int = 270
    UNLABELED_SPLIT_SIZE: int = 540
    SPLIT_SEED: int = 42
    IMAGE_RESIZE: int = 256
    IMAGE_SIZE: int = 224
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = torch.cuda.is_available()
    USE_WEIGHTED_SAMPLER: bool = False

    # Semi-supervised settings (FixMatch)
    FIXMATCH_TAU: float = 0.85
    FIXMATCH_LAMBDA_U: float = 0.5
    FIXMATCH_RAMPUP_EPOCHS: int = 10
    FIXMATCH_MIN_TAU: float = 0.4
    FIXMATCH_USE_CLASS_THRESHOLDS: bool = True
    FIXMATCH_DISTRIBUTION_ALIGNMENT: bool = True
    FIXMATCH_DA_MOMENTUM: float = 0.9
    FIXMATCH_SHARPEN_T: float = 1.0
    FLEXMATCH_ENABLE: bool = True
    FLEXMATCH_MOMENTUM: float = 0.7

    # Training control
    EARLY_STOP_PATIENCE: int = 5
    SAVE_BEST_CHECKPOINT: bool = True
    BEST_METRIC: str = "val_auc"  # "val_auc" or "val_ap"
    CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    WARMUP_EPOCHS: int = 2
    USE_AMP: bool = True
    MAX_GRAD_NORM: float = 1.0

    # Derived stats (populated at runtime)
    TRAIN_POS_RATIO: float | None = None
    TRAIN_NEG_RATIO: float | None = None


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
