from pathlib import Path

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
CLASS_NAMES = ["def_front", "ok_front"]
CLASS_DISPLAY_NAMES = {
    "def_front": "Defective",
    "ok_front": "Non-defective / OK",
}
DEFAULT_MODEL_PATH = Path("models/quality_inspection_model.keras")
DEFAULT_OUTPUT_DIR = Path("outputs")
