import torch
import pandas as pd
import cv2
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("❌ Usage: python detect_and_save_csv.py <video_filename>")
    exit()

video_filename = sys.argv[1]
video_path = Path("yolov5/output/exp") / video_filename
video_name = Path(video_filename).stem

# Path to save CSV in yolo_in_csv directory
output_csv_path = Path("R&D/output/yolo_in_csv") / f"{video_name}.csv"
output_csv_path.parent.mkdir(parents=True, exist_ok=True)

# Path to save annotated video in yolov5/output/exp
annotated_video_path = Path("yolov5/output/exp") / f"{video_name}_annotated.mp4"

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.eval()

# Open the input video
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print("❌ Error: Cannot open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer for annotated output
out = cv2.VideoWriter(
    str(annotated_video_path),
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

frame_number = 0
all_detections = []

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    detections = results.pandas().xyxy[0]
    detections['time'] = frame_number / fps
    detections['frame'] = frame_number
    all_detections.append(detections)

    # Draw detection boxes and labels
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = f"{row['name']} {row['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    out.write(frame)
    frame_number += 1

# Finalize everything
cap.release()
out.release()
cv2.destroyAllWindows()

# Save CSV file
if all_detections:
    final_detections = pd.concat(all_detections, ignore_index=True)
    final_detections.to_csv(output_csv_path, index=False)
    print(f"✅ CSV saved to: {output_csv_path}")
    print(f"✅ Annotated video saved to: {annotated_video_path}")
else:
    print("⚠️ No detections found. CSV not created.")
