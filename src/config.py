import os
from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

# Audio parameters
SAMPLE_RATE = 16000
N_MFCC = 13
MAX_AUDIO_LENGTH = 10  # seconds

# Text parameters
MAX_TEXT_LENGTH = 512
VOCAB_SIZE = 30000

# Model parameters
HIDDEN_DIM = 128
DROPOUT = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 