"""Live webcam backpack scanner using OpenCV."""

import sys

import cv2

from config import CONFIDENCE_THRESHOLD, MODEL_PATH, WEBCAM_INDEX
from scanner import (
    detect_items,
    draw_status_overlay,
    find_missing_items,
    get_current_day,
    get_required_items,
    load_model,
    open_webcam,
)


def main() -> int:
    if not MODEL_PATH.exists():
        print(
            f"Error: Model not found at {MODEL_PATH}\n"
            "Train your model (see train.py / README) and place best.pt in this folder."
        )
        return 1

    print("Loading model...")
    model = load_model()
    print(f"Model classes: {list(model.names.values())}")

    day = get_current_day()
    required = get_required_items(day)
    print(f"Today is {day}. Required: {', '.join(required)}")
    print("Press 'q' to quit.\n")

    cap = open_webcam(WEBCAM_INDEX)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam.")
                break

            detected, frame = detect_items(frame, model, CONFIDENCE_THRESHOLD)
            missing = find_missing_items(required, detected)
            draw_status_overlay(frame, day, required, missing)

            cv2.imshow("Before-You-Leave Backpack Scanner", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
