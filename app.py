import streamlit as st
import pandas as pd

from pathlib import Path
import sys

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


sys.path.append(str(Path(__file__).resolve().parent))

from main import run_pipeline

st.set_page_config(page_title="Video Processing App", layout="centered")
st.title(" Video Processing Pipeline 🎥")
st.markdown("Select a video and a blur level, and the system will run all the steps and return the result.")

# --- Input video selection ---
# --- Input video upload ---
st.markdown("### Upload a video (MP4)")
uploaded_file = st.file_uploader("Choose a .mp4 file", type=["mp4"])

video_file = None
if uploaded_file is not None:
    inputs_dir = Path("inputs")
    inputs_dir.mkdir(exist_ok=True)
    video_file = inputs_dir / uploaded_file.name
    with open(video_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded successfully: {uploaded_file.name}")



# --- Blur level selection ---
blur_options = {
    "Vision of an adult - No blur": 0,
    "Vision of a two-year-old baby - Light blur": 5,
    "Vision of a one-year-old baby - Medium blur": 10,
    "Vision of a six-month-old baby - Strong blur": 20

}
blur_choice = st.selectbox("Select blur level:", list(blur_options.keys()))
blur_level = blur_options[blur_choice]


# --- Start button ---
if st.button(" Run ▶️ "):
    st.info("Running video processing...")
    try:
        detection_dir, analysis_path = run_pipeline(video_file, blur_level)
       # st.success(f"✅ Pipeline completed! Outputs saved in:\n- {detection_dir}\n- {analysis_path}")
       #st.info("Done")

        video_stem = Path(video_file).stem

        # --- Detected video ---
        if detection_dir and isinstance(detection_dir, (str, Path)) and Path(detection_dir).exists():
            video_stem = Path(video_file).stem
            #detected_path = Path(detection_dir) / f"{video_stem}_detected.mp4"
            # Try to find the detected video that matches the stem
            detected_path = None
            for file in Path(detection_dir).glob(f"{video_stem}*detected.mp4"):
                detected_path = file
                break

            if detected_path.exists():
                st.markdown("### 🧠 YOLOv5 Detected Video")
                st.video(str(detected_path))
            else:
                st.warning(f"⚠️ Detected video not found: {detected_path}")
        else:
            st.warning("⚠️ Detection path not returned correctly.")


        # --- CSV Output ---
        if analysis_path and Path(analysis_path).exists():
            st.markdown("### 📊 Confidence CSV Output")
            st.download_button(
                label="⬇️ Download CSV",
                data=open(analysis_path, "rb").read(),
                file_name=Path(analysis_path).name,
                mime="text/csv"
            )
            st.dataframe(pd.read_csv(analysis_path))
        else:
            st.warning("⚠️ Confidence analysis CSV not found.")

    except Exception as e:
        st.error(f"❌ Pipeline failed: {e}")



