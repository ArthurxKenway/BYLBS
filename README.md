# Before-You-Leave Backpack Scanner

YOLO-powered desk scanner that checks your webcam for school supplies against a daily timetable.

## How it works

1. **Scanner (YOLO)** — detects stationery on your desk from the webcam
2. **Logic (timetable)** — defines what you need each weekday
3. **Dashboard** — compares lists and shows Ready / Missing status

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Step 1: Get a dataset

1. Go to [Roboflow Universe](https://universe.roboflow.com)
2. Search for **School Supplies** or **Stationery Detection**
3. Download in **YOLOv8** format
4. Unzip the folder

## Step 2: Train the model

**Google Colab** (free GPU):

```python
!pip install ultralytics
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="path/to/data.yaml", epochs=50, imgsz=640, device=0)
```

Download `runs/detect/train/weights/best.pt` and place it in this project folder.

**Or locally:**

```bash
python train.py --data path/to/data.yaml --epochs 50 --device 0
```

After training, update `config.py` → `TIMETABLE` so item names match your dataset labels (comparison is case-insensitive).

## Step 3: Run the app

**Live webcam (OpenCV):**

```bash
python app.py
```

Press `q` to quit.

**Web dashboard (Streamlit):**

```bash
streamlit run app_streamlit.py
```

Use the sidebar to simulate different days of the week.

## Project layout

```
YoLo/
├── app.py              # Live OpenCV scanner
├── app_streamlit.py    # Streamlit web dashboard
├── scanner.py          # Shared detection + checklist logic
├── config.py           # Timetable and settings
├── train.py            # Local training script
├── best.pt             # Your trained weights (not in git)
└── requirements.txt
```
