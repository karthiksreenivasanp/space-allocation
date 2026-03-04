"""
ParkXplore – Car Detection using RT-DETR
Detects cars in test images and saves annotated results to ./output/
"""
import cv2
import os
import glob
from ultralytics import RTDETR

# ── Config ───────────────────────────────────────────────────────────
MODEL_PATH  = 'runs/parkxplore_transformer/weights/best.pt'
SOURCE_PATH = 'dataset/test/images'
OUTPUT_DIR  = 'output'
CONF        = 0.4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model ───────────────────────────────────────────────────────
print("=" * 55)
print("  ParkXplore – RT-DETR Car Detection")
print("=" * 55)
print(f"  Model  : {MODEL_PATH}")
print(f"  Source : {SOURCE_PATH}")
print(f"  Output : {OUTPUT_DIR}/")
print("=" * 55)

print("\nLoading model...")
model = RTDETR(MODEL_PATH)
print("Model loaded ✅\n")

# ── Run inference ────────────────────────────────────────────────────
images = sorted(glob.glob(os.path.join(SOURCE_PATH, '*.jpg')) +
                glob.glob(os.path.join(SOURCE_PATH, '*.png')))

if not images:
    print(f"❌  No images found in {SOURCE_PATH}")
    exit(1)

print(f"Found {len(images)} test images.\n")

total_cars = 0

for idx, img_file in enumerate(images):
    frame = cv2.imread(img_file)
    if frame is None:
        print(f"  [{idx+1}] ⚠️  Could not read {os.path.basename(img_file)}")
        continue

    # Detect
    results = model.predict(frame, conf=CONF, verbose=False)

    # Draw bounding boxes
    num_cars = len(results[0].boxes)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf_score = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"Car {conf_score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 200), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Summary overlay
    summary = f"Cars detected: {num_cars}"
    cv2.rectangle(frame, (0, 0), (len(summary) * 12 + 10, 38), (30, 30, 30), -1)
    cv2.putText(frame, summary, (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"result_{os.path.basename(img_file)}")
    cv2.imwrite(out_path, frame)

    total_cars += num_cars
    print(f"  [{idx+1:02d}/{len(images)}] {os.path.basename(img_file):45s}  Cars={num_cars}  → saved")

print("\n" + "=" * 55)
print(f"  ✅  Done! Processed {len(images)} images")
print(f"  🚗  Total cars detected : {total_cars}")
print(f"  📁  Results saved to   : {os.path.abspath(OUTPUT_DIR)}/")
print("=" * 55)
