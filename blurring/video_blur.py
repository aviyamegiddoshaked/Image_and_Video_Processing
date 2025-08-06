import cv2
from PIL import Image, ImageFilter
import numpy as np
from pathlib import Path  # ✅ חשוב!
import os


def find_video_file(video_name: str):
    inputs_dir = Path("inputs")  # path relative to where main.py is run
    direct_path = inputs_dir / video_name

    # Try as-is
    if direct_path.is_file():
        return direct_path

    # Try with common extensions
    for ext in [".mp4", ".mov", ".avi", ".mkv"]:
        alt_path = inputs_dir / (video_name + ext)
        if alt_path.is_file():
            return alt_path

    return None


def blur_video(video_name: str, blur_level: int):
    input_path = Path(video_name)

    if not input_path.exists():
        input_path = find_video_file(video_name)
        if input_path is None:
            print(f"❌ Error: Video file not found in inputs/: {video_name}")
            return None

    if blur_level == 0:
        return str(input_path)

    output_dir = Path("blurring") / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"{input_path.stem}_blur{blur_level}.mp4"
    output_path = output_dir / output_name

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print("❌ Error: Cannot open video file.")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 25  # Fallback fps value if not available

    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        blurred = cv2.GaussianBlur(frame, (blur_level | 1, blur_level | 1), 0)
        out.write(blurred)

    cap.release()
    out.release()

    return str(output_path)


