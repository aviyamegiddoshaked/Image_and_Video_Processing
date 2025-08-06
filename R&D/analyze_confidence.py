import pandas as pd
from pathlib import Path

def analyze_confidence(csv_path: str):
    print("🚀 Entered analyze_confidence function!")
    """
    Analyzes YOLOv5 detection confidence values in a given CSV file.

    Args:
        csv_path (str): Path to the YOLOv5-generated CSV file.

    Performs:
        - Loads the CSV and checks for the 'confidence' column.
        - Calculates average confidence and total detections.
        - Saves the results into R&D/output/yolo_analysis/ as a new CSV.
        - Prints results to the terminal.
    """
    print(f"📁 Analyzing file: {csv_path}")
    print(f"🔽 Will save to: R&D/output/yolo_analysis/{Path(csv_path).stem}_stats.csv")

    path = Path(csv_path)
    if not path.exists():
        print(f"❌ File not found: {csv_path}")
        return

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"❌ Failed to read CSV file: {e}")
        return

    if "confidence" not in df.columns:
        print("⚠️ Column 'confidence' not found in CSV.")
        return

    # Extract metrics
    video_name = path.stem
    avg_conf = df["confidence"].mean()
    total_detections = len(df)

    # Print results
    print(f"📊 Analysis for video: {video_name}")
    print(f"🔹 Average confidence: {avg_conf:.4f}")
    print(f"🔹 Total detections: {total_detections}")

    # Prepare output path
    output_dir = Path("R&D/output/yolo_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_name}_stats.csv"

    # Save results
    stats_df = pd.DataFrame([{
        "video": video_name,
        "avg_confidence": avg_conf,
        "total_detections": total_detections
    }])
    try:
        stats_df.to_csv(output_path, index=False)
        print(f"✅ Analysis results saved to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save analysis results: {e}")
