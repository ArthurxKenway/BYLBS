"""Core detection and checklist logic."""

from __future__ import annotations

import datetime
from functools import lru_cache

import cv2
from ultralytics import YOLO

from config import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_DAY,
    MODEL_PATH,
    TIMETABLE,
    WEEKDAYS,
)


def normalize_item_name(name: str) -> str:
    return name.strip().lower().replace("_", " ")


@lru_cache(maxsize=1)
def load_model(model_path: str | None = None) -> YOLO:
    path = model_path or str(MODEL_PATH)
    return YOLO(path)


def get_current_day() -> str:
    day = datetime.datetime.now().strftime("%A")
    return day if day in TIMETABLE else DEFAULT_DAY


def get_required_items(day: str | None = None) -> list[str]:
    selected_day = day or get_current_day()
    if selected_day not in TIMETABLE:
        selected_day = DEFAULT_DAY
    return TIMETABLE[selected_day]


def detect_items(
    frame,
    model: YOLO,
    confidence: float = CONFIDENCE_THRESHOLD,
) -> tuple[set[str], object]:
    """Run YOLO on a frame and return detected class names plus annotated frame."""
    results = model(frame, conf=confidence, verbose=False)
    detected: set[str] = set()
    annotated = frame

    for result in results:
        annotated = result.plot()
        for box in result.boxes:
            class_id = int(box.cls[0])
            detected.add(model.names[class_id])

    return detected, annotated


def find_missing_items(required: list[str], detected: set[str]) -> list[str]:
    detected_normalized = {normalize_item_name(item) for item in detected}
    return [
        item
        for item in required
        if normalize_item_name(item) not in detected_normalized
    ]


def is_ready(required: list[str], detected: set[str]) -> bool:
    return len(find_missing_items(required, detected)) == 0


def open_webcam(index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {index}. "
            "Check that a camera is connected and not in use by another app."
        )
    return cap


def draw_status_overlay(
    frame,
    day: str,
    required: list[str],
    missing: list[str],
) -> None:
    cv2.putText(
        frame,
        f"Today is: {day}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Required: {', '.join(required)}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    if not missing:
        cv2.putText(
            frame,
            "STATUS: READY TO GO! :)",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            3,
        )
    else:
        cv2.putText(
            frame,
            f"MISSING: {', '.join(missing)}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3,
        )
