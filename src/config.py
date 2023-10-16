
from pathlib import Path

# =========================
# Folders
# =========================

ROOT = Path(__file__).absolute().parent.parent

WEIGHTS_FOLDER = ROOT.joinpath('weights')
SRC_FOLDER = ROOT.joinpath('src')
CONFIGS_FOLDER = ROOT.joinpath('configs')
DATASET_FOLDER = ROOT.joinpath('datasets')


# =========================
# Logger
# =========================

# CRITICAL: 50
# ERROR: 40
# WARNING: 30
# INFO: 20
# DEBUG: 10
# NOTSET: 0
LOG_LEVEL = 20  # Ignore logging messages which are less severe
if LOG_LEVEL < 20:
    LOG_FORMAT = '%(levelname)s %(filename)s (%(lineno)d) : %(message)s'  # Logging messages string format (for development)
else:
    LOG_FORMAT = '%(levelname)s : %(message)s'  # Logging messages string format (for release)
LOG_FILENAME = None  # Save logging messages to file. None: console


# =========================
# YOLOv8
# =========================

SUPPORTED_MODELS = (
    'YOLOv8n',
    'YOLOv8s',
    'YOLOv8m',
    'YOLOv8l',
    'YOLOv8x'
)

SUPPORTED_DETECTION_MODELS = {model: model.lower() + '.pt' for model in SUPPORTED_MODELS}

SUPPORTED_CLASSIFICATION_MODELS = {model: model.lower() + '-cls.pt' for model in SUPPORTED_MODELS}

SUPPORTED_OPTIMIZERS = (
    'SGD',
    'Adam',
    'Adamax',
    'AdamW',
    'NAdam',
    'RAdam',
    'RMSProp',
    'auto'
)
