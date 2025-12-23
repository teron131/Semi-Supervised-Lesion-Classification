"""Constants used throughout the codebase."""

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Image file extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Label values
LABEL_BENIGN = 0.0
LABEL_MALIGNANT = 1.0

# Class names
CLASS_BENIGN = "benign"
CLASS_MALIGNANT = "malignant"

# Rampup function constants
RAMPUP_EXPONENT = 5.0

# Numerical stability constants
EPSILON = 1e-6
EPSILON_8 = 1e-8

# Distribution alignment bounds
DA_PROB_MIN = 1e-6
DA_PROB_MAX = 1.0 - 1e-6
