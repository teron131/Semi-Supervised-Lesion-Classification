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
    LEARNING_RATE: float = 1e-4  # Lower LR for more stable training
    WEIGHT_DECAY: float = 1e-2  # Stronger regularization to combat overfitting

    # Model Settings
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    PRE_TRAINED: bool = True
    NUM_CLASSES: int = 1

    # Loss Settings
    FOCAL_GAMMA: float = 2.0
    FOCAL_ALPHA: float = 0.80  # Slightly adjusted to be less aggressive than 0.85
    SUPERVISED_LOSS: str = "focal"  # Try focal loss with corrected alpha
    POS_WEIGHT: float = 2.5
    AUTO_POS_WEIGHT: bool = True  # Auto-compute based on class ratio
    POS_WEIGHT_MAX: float = 4.0  # Allow higher weight for minority class

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
    FIXMATCH_TAU: float = 0.85  # Lowered from 0.95 to accept more pseudo-labels
    FIXMATCH_TAU_POS: float = 0.75  # Lower for minority class (malignant)
    FIXMATCH_TAU_NEG: float = 0.90  # Higher for majority class (benign)
    FIXMATCH_USE_ASYMMETRIC_TAU: bool = True  # Enable class-specific thresholds
    FIXMATCH_TAU_SCHEDULE: bool = False
    FIXMATCH_TAU_START: float = 0.95
    FIXMATCH_TAU_END: float = 0.85
    FIXMATCH_TAU_SCHEDULE_EPOCHS: int = 8
    FIXMATCH_LAMBDA_U: float = 2.0  # Increased to leverage unlabeled data more
    FIXMATCH_RAMPUP_EPOCHS: int = 10  # Slower ramp-up for stability
    FIXMATCH_MIN_TAU: float = 0.75  # Allow lower thresholds for FlexMatch
    FIXMATCH_USE_CLASS_THRESHOLDS: bool = False
    FIXMATCH_DISTRIBUTION_ALIGNMENT: bool = True  # Enable to correct class bias
    FIXMATCH_DA_MOMENTUM: float = 0.9
    FIXMATCH_SHARPEN_T: float = 0.5  # Sharpen pseudo-labels for harder targets
    FLEXMATCH_ENABLE: bool = True  # Enable class-adaptive thresholds
    FLEXMATCH_MOMENTUM: float = 0.9  # Increased for more stable threshold updates
    SOFT_PSEUDO_LABELS: bool = True  # Weight by confidence above threshold
    FLEXMATCH_WARMUP_EPOCHS: int = 3  # Earlier FlexMatch activation
    FLEXMATCH_TAU_MIN: float = 0.75  # Allow thresholds to drop lower
    FIXMATCH_USE_TOPK: bool = False
    FIXMATCH_TOPK_POS: int = 8
    FIXMATCH_TOPK_NEG: int = 16

    # MixUp augmentation (creates synthetic samples)
    MIXUP_ENABLE: bool = True
    MIXUP_ALPHA: float = 0.4  # Beta distribution parameter (higher = more mixing)
    MIXUP_PROB: float = 0.5  # Probability of applying MixUp per batch

    # Training control
    EARLY_STOP_PATIENCE: int = 10  # More patience for SSL to fully converge
    SAVE_BEST_CHECKPOINT: bool = True
    BEST_METRIC: str = "val_ap"  # AP is better for imbalanced data
    CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    WARMUP_EPOCHS: int = 2
    USE_AMP: bool = True
    MAX_GRAD_NORM: float = 1.0
    EMA_ENABLE: bool = True
    EMA_DECAY: float = 0.999
    FREEZE_BACKBONE_EPOCHS: int = 2
    LR_LAYER_DECAY: float = 0.8
    INIT_BIAS_FROM_PRIOR: bool = True  # Initialize output bias from class distribution

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
