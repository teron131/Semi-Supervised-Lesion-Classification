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
    EPOCHS: int = 40  # Slightly more epochs for better SSL convergence
    LEARNING_RATE: float = 5e-5  # Lower LR for more stable training
    WEIGHT_DECAY: float = 3e-2  # Stronger regularization to combat overfitting
    HEAD_LR_MULT: float = 3.0  # Faster adaptation for the classification head

    # Model Settings
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    PRE_TRAINED: bool = True
    NUM_CLASSES: int = 1

    # Loss Settings
    FOCAL_GAMMA: float = 2.0
    FOCAL_ALPHA: float = 0.5
    AUTO_FOCAL_ALPHA: bool = False
    SUPERVISED_LOSS: str = "focal"
    POS_WEIGHT: float = 2.0
    AUTO_POS_WEIGHT: bool = True
    POS_WEIGHT_MAX: float = 10.0

    # Data settings
    TRAIN_SPLIT_SIZE: int = 270
    UNLABELED_SPLIT_SIZE: int = 540
    SPLIT_SEED: int = 42
    IMAGE_RESIZE: int = 256
    IMAGE_SIZE: int = 224
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = torch.cuda.is_available()
    USE_WEIGHTED_SAMPLER: bool = True

    # Semi-supervised settings (FixMatch)
    FIXMATCH_TAU: float = 0.80  # Lowered slightly
    FIXMATCH_TAU_POS: float = 0.70  # Increased slightly for better precision
    FIXMATCH_TAU_NEG: float = 0.95  # Increased for more certain benign labels
    FIXMATCH_USE_ASYMMETRIC_TAU: bool = True
    FIXMATCH_TAU_SCHEDULE: bool = True  # Enable schedule
    FIXMATCH_TAU_START: float = 0.70
    FIXMATCH_TAU_END: float = 0.85
    FIXMATCH_TAU_SCHEDULE_EPOCHS: int = 15
    FIXMATCH_LAMBDA_U: float = 2.0  # Conservative weight
    FIXMATCH_RAMPUP_EPOCHS: int = 5  # Faster rampup
    FIXMATCH_MIN_TAU: float = 0.60
    FIXMATCH_USE_CLASS_THRESHOLDS: bool = True
    FIXMATCH_DISTRIBUTION_ALIGNMENT: bool = True
    FIXMATCH_DA_MOMENTUM: float = 0.9
    FIXMATCH_SHARPEN_T: float = 0.5
    FLEXMATCH_ENABLE: bool = True
    FLEXMATCH_MOMENTUM: float = 0.9
    SOFT_PSEUDO_LABELS: bool = True
    FLEXMATCH_WARMUP_EPOCHS: int = 3
    FLEXMATCH_TAU_MIN: float = 0.60
    FIXMATCH_USE_TOPK: bool = False
    FIXMATCH_TOPK_POS: int = 8
    FIXMATCH_TOPK_NEG: int = 16
    FIXMATCH_MU: int = 3  # Ratio of unlabeled to labeled batch size

    # MixUp augmentation (creates synthetic samples)
    MIXUP_ENABLE: bool = True
    MIXUP_ALPHA: float = 0.4
    MIXUP_PROB: float = 0.5

    # Training control
    TRAIN_STEPS_PER_EPOCH: int = 64  # Define epoch length in batches
    EARLY_STOP_PATIENCE: int = 10
    EARLY_STOP_MIN_DELTA: float = 1e-4
    EARLY_STOP_METRIC: str | None = "val_auc"
    SAVE_BEST_CHECKPOINT: bool = True
    BEST_METRIC: str = "val_auc"
    CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    WARMUP_EPOCHS: int = 2
    USE_AMP: bool = True
    MAX_GRAD_NORM: float = 1.0
    EMA_ENABLE: bool = True
    EMA_DECAY: float = 0.999
    FREEZE_BACKBONE_EPOCHS: int = 5
    LR_LAYER_DECAY: float = 0.8
    INIT_BIAS_FROM_PRIOR: bool = True

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
