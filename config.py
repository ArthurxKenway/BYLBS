"""Shared configuration for the backpack scanner."""

from pathlib import Path

ROOT_DIR = Path(__file__).parent
MODEL_PATH = ROOT_DIR / "best.pt"


DATASET_CLASS_NAMES = [
    "book",
    "correction_pen",
    "eraser",
    "marker",
    "notebook",
    "pen",
    "pencil",
    "scissor",
    "sharpener",
    "stapler",
    "stationery_set",
]


TIMETABLE = {
    "Monday": ["notebook", "pen", "pencil"],
    "Tuesday": ["notebook", "book", "eraser"],
    "Wednesday": ["stationery_set", "marker", "sharpener"],
    "Thursday": ["notebook", "pencil", "stapler"],
    "Friday": ["notebook", "pen", "book", "stationery_set"],
}

WEEKDAYS = list(TIMETABLE.keys())
DEFAULT_DAY = "Monday"

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
WEBCAM_INDEX = 0
