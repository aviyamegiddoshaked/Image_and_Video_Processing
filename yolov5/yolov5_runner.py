from yolov5.detect import run
from pathlib import Path
import shutil
import importlib.util
import sys

# Load analyze_confidence.py from R&D
rnd_path = Path(__file__).resolve().parents[1] / "R&D"
sys.path.append(str(rnd_path))
spec = importlib.util.spec_from_file_location("analyze_confidence", rnd_path / "analyze_confidence.py")
analyze_confidence_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_confidence_module)


def run_detection(video_path: str, weights_path='yolov5s.pt'):
    yolov5_output_dir = Path("yolov5/output")
    yolov5_output_dir.mkdir(parents=True, exist_ok=True)
    video_stem = Path(video_path).stem
    exp_dir = yolov5_output_dir / "exp"
    csv_output_dir = Path("R&D/output/yolo_in_csv")
    csv_output_dir.mkdir(parents=True, exist_ok=True)

    print("🔎 Running YOLOv5 detection...")
    run(
        weights=weights_path,
        source=video_path,
        project=str(yolov5_output_dir),
        name='exp',
        exist_ok=True,
        save_csv=True
    )

    # Rename output video
    raw_output_video = exp_dir / f"{video_stem}.mp4"
    final_output_video = exp_dir / f"{video_stem}_detected.mp4"
    if raw_output_video.exists():
        try:
            raw_output_video.rename(final_output_video)
            print(f"✅ Renamed output video to: {final_output_video}")
        except Exception as e:
            print(f"⚠️ Failed to rename video: {e}")
    else:
        print("⚠️ Detection video not found.")

    # Move CSV file
    detected_csv = exp_dir / f"{video_stem}_detected.csv"
    final_csv_path = csv_output_dir / f"{video_stem}.csv"

    if detected_csv.exists():
        try:
            if final_csv_path.exists():
                final_csv_path.unlink()
            shutil.move(str(detected_csv), str(final_csv_path))
            print(f"✅ CSV saved to: {final_csv_path}")
        except Exception as e:
            print(f"⚠️ Failed to move CSV file: {e}")
    else:
        print("❌ CSV file not found after detection.")

    # Run analysis (optional)
    print("📊 Running confidence analysis...")
    try:
        analyze_confidence_module.analyze_confidence(str(final_csv_path))
        print("✅ analyze_confidence() completed successfully.")
    except Exception as e:
        print(f"❌ analyze_confidence() failed: {e}")

    # Always return detection path
    return str(exp_dir)
