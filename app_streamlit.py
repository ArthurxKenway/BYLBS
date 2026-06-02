"""Web dashboard for the backpack scanner."""

import streamlit as st
from PIL import Image
import numpy as np

from config import CONFIDENCE_THRESHOLD, MODEL_PATH, WEEKDAYS
from scanner import (
    detect_items,
    find_missing_items,
    get_current_day,
    get_required_items,
    load_model,
)

st.set_page_config(
    page_title="Backpack Scanner",
    page_icon="🎒",
    layout="wide",
)

st.title("Before-You-Leave Backpack Scanner")
st.caption("YOLO-powered desk check, make sure you have everything before you go.")


@st.cache_resource
def get_model():
    return load_model()


def main() -> None:
    if not MODEL_PATH.exists():
        st.error(
            f"**Model not found** at `{MODEL_PATH}`\n\n"
            "Train your YOLO model and save `best.pt` in the project folder. "
            "See README for steps."
        )
        st.stop()

    model = get_model()

    with st.sidebar:
        st.header("Settings")
        default_day = get_current_day()
        day_index = WEEKDAYS.index(default_day) if default_day in WEEKDAYS else 0
        selected_day = st.selectbox(
            "Day of the week",
            WEEKDAYS,
            index=day_index,
            help="Simulate a different day to test the checklist.",
        )
        confidence = st.slider(
            "Detection confidence",
            min_value=0.1,
            max_value=0.95,
            value=CONFIDENCE_THRESHOLD,
            step=0.05,
        )
        st.divider()
        st.subheader("Today's checklist")
        required = get_required_items(selected_day)
        for item in required:
            st.write(f"- {item}")

    col_feed, col_status = st.columns([2, 1])

    with col_feed:
        st.subheader("Scan your desk")
        camera_image = st.camera_input("Point your camera at your desk items")

    with col_status:
        st.subheader("Status")

        if camera_image is None:
            st.info("Enable your camera above to start scanning.")
            return

        image = Image.open(camera_image)
        frame = np.array(image.convert("RGB"))

        detected, annotated = detect_items(frame, model, confidence)
        missing = find_missing_items(required, detected)

        st.image(
            annotated,
            caption="Detected items (bounding boxes)",
            use_container_width=True,
        )

        st.markdown(f"**Day:** {selected_day}")
        st.markdown(f"**Detected:** {', '.join(sorted(detected)) or 'None'}")

        if not missing:
            st.success("Ready to Go! All required items detected.")
        else:
            st.error(f"Missing Items: {', '.join(missing)}")

        with st.expander("Model class labels"):
            st.write(list(model.names.values()))


if __name__ == "__main__":
    main()
