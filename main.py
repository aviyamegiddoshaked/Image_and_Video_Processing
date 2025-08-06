from blurring.video_blur import blur_video
from yolov5.yolov5_runner import run_detection
from pathlib import Path


def run_pipeline(video_file, blur_level):
    # Ensure input is Path
    video_name = Path(video_file.name) if hasattr(video_file, "name") else Path(video_file)

    # Save file to inputs/ if it's an uploaded file
    input_path = Path("inputs") / video_name.name
    if hasattr(video_file, "read"):
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with open(input_path, "wb") as f:
            f.write(video_file.read())
    else:
        input_path = Path(video_file)

    # Run blurring
    output_path = blur_video(str(input_path), blur_level)
    if output_path is None:
        raise RuntimeError("Blurring failed.")

    # Run YOLO detection
    detection_dir = run_detection(output_path)

    # Check if analysis CSV was created
    analysis_path = Path("R&D/output/yolo_analysis") / f"{Path(output_path).stem}_stats.csv"
    if analysis_path.exists():
        print(f"✅ Confidence analysis saved to: {analysis_path}")
        return str(detection_dir), str(analysis_path)
    else:
        print("⚠️ Analysis CSV not found.")
        return str(detection_dir), None


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <video_name>")
        print("Example: python main.py hello or hello.mp4")
        return

    video_name = sys.argv[1]

    print("Choose blur level:")
    print("0 = no blur")
    print("5 = light blur")
    print("10 = medium blur")
    print("20 = strong blur")
    blur_input = input("Enter blur level (0, 5, 10, 20): ")

    try:
        blur_level = int(blur_input)
        assert blur_level in [0, 5, 10, 20]
    except:
        print("❌ Invalid blur level. Please enter 0, 5, 10, or 20.")
        return

    output_path = blur_video(video_name, blur_level)
    if output_path is None:
        print("❌ Blurring failed.")
    else:
        print(f"✅ Blurring successful. Output saved to: {output_path}")

    detection_dir = run_detection(output_path)
    print(f"✅ Detection completed. Results saved to: {detection_dir}")


if __name__ == "__main__":
    main()
