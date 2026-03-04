# ParkXplore – Intelligent Parking Space Detection 🚗🅿️

ParkXplore is a computer vision project designed to detect and manage parking slot occupancy in real-time. It uses the state-of-the-art **RT-DETR** (Real-Time Detection Transformer) model to identify cars, combined with an interactive point-based slot marking system to determine if specific parking spaces are **Occupied** or **Free**.

---

## 🌟 Features

- **High-Accuracy Detection:** Leverages RT-DETR fine-tuned on the `car-parking-6effs` dataset.
- **Interactive Marking Tool:** Easily mark valid parking slots on any reference image via a simple point-and-click GUI.
- **Automated Occupancy Checks:** Analyzes bounding boxes and marked points to classify slots as 🔴 Occupied or 🟢 Free.
- **Batch Processing:** Automatically processes entire folders of test images and outputs annotated results with summary overlays.

---

## 🛠️ Setup & Installation

**1. Clone the repository:**
```bash
git clone https://github.com/karthiksreenivasanp/space-allocation.git
cd space-allocation
```

**2. Create and activate a Virtual Environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

*(Note: Depending on your OS, you may need to install `python3-opencv` globally or ensure your OpenCV backend works with your display server).*

---

## 🚀 Usage Guide

The system operates in a simple two-step process:

### Step 1: Mark Parking Slots
Open a reference image and click to define the center point of each parking space.

```bash
python mark_slots.py
```
**Controls:**
- **Left-Click:** Place a slot point
- **'u':** Undo the last point
- **'s':** Save all points to `parking_slots.json` and exit
- **ESC:** Quit without saving

### Step 2: Run Occupancy Detection
Run the detector to analyze all test images in the `dataset/test/images/` folder. It will use the points saved from Step 1.

```bash
python check_occupancy.py
```
- The script detects cars using RT-DETR.
- If a car's bounding box covers a marked point, the slot is labeled **Occupied (Red)**.
- If the point is clear, it is labeled **Free (Green)**.
- Annotated output images are saved in the `output/` directory with a visual summary bar.

---

## 📁 Project Structure

```text
├── dataset/                     # Training, validation, and test images/labels
├── runs/                        # Model training runs and weights (best.pt)
├── output/                      # Annotated result images from the occupancy checker
├── check_occupancy.py           # Core logic for batch occupancy detection
├── mark_slots.py                # Interactive point-and-click GUI for defining slots
├── parking_slots.json           # Saved coordinates of the marked parking slots
├── requirements.txt             # Python package dependencies
└── README.md                    # Project documentation
```

---

## 🧠 Model Details

- **Architecture:** RT-DETR (Transformer-based object detection)
- **Classes:** 1 (`Car`)
- **Confidence Threshold:** `0.4` (configurable via `--conf`)

*Developed by Karthik Sreenivasan P.*
