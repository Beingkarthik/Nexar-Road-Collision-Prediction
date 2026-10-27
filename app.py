import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "improved_resnet18.pth"

st.set_page_config(
    page_title="Nexar Collision Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚘 Road Collision & Near-Miss Analysis")
st.markdown(
    """
    **AI-powered video analysis using a trained ResNet18 model**

    Review frame-level risk scores, inspect flagged moments, and export
    an analysis report for research and demonstration.
    """
)
st.divider()

st.warning(
    "Research demonstration only. Scores are not calibrated collision "
    "probabilities and must not be used as a real-time driving safety system."
)


@st.cache_resource
def load_model(path):
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. "
            "Copy your actual improved_resnet18.pth file into the models folder."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 2)
    )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, device


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


st.sidebar.header("Settings")

threshold = st.sidebar.slider(
    "Positive-score threshold",
    0.10, 0.95, 0.50, 0.05
)

frame_step = st.sidebar.number_input(
    "Analyze every Nth frame",
    min_value=1,
    max_value=300,
    value=10
)

max_frames = st.sidebar.number_input(
    "Maximum frames to analyze (0 = all)",
    min_value=0,
    value=500,
    step=100
)


def analyze_video(video_path, model, device):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open the video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions.")

    output_file = tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False
    )
    output_path = output_file.name
    output_file.close()

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    results = []
    frame_number = 0
    analyzed = 0

    progress = st.progress(0.0)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if (
                frame_number % int(frame_step) == 0
                and (int(max_frames) == 0 or analyzed < int(max_frames))
            ):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                tensor = preprocess(image).unsqueeze(0).to(device)

                with torch.inference_mode():
                    logits = model(tensor)
                    probabilities = torch.softmax(logits, dim=1)[0]

                score = float(probabilities[1].item())
                exceeds = score >= threshold

                results.append({
                    "frame_number": frame_number,
                    "timestamp_seconds": frame_number / fps,
                    "positive_class_score": score,
                    "threshold_exceeded": exceeds,
                    "status": (
                        "Threshold exceeded" if exceeds else "Below threshold"
                    )
                })

                analyzed += 1

            if results:
                latest = results[-1]
                score = latest["positive_class_score"]
                color = (0, 0, 255) if score >= threshold else (0, 180, 0)

                cv2.putText(
                    frame,
                    f"Positive-class score: {score:.3f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    latest["status"],
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

            writer.write(frame)
            frame_number += 1

            if total_frames > 0:
                progress.progress(min(frame_number / total_frames, 1.0))

    finally:
        cap.release()
        writer.release()
        progress.empty()

    if not results:
        os.remove(output_path)
        raise RuntimeError("No frames were analyzed.")

    return pd.DataFrame(results), output_path, fps, total_frames



# --- CONNECTED SAMPLE VIDEO ANALYSIS ---
SAMPLE_VIDEO_DIR = BASE_DIR / "demo" / "videos"
sample_videos = sorted(SAMPLE_VIDEO_DIR.glob("*.mp4")) if SAMPLE_VIDEO_DIR.exists() else []

st.subheader("Choose a road video")
source_mode = st.radio(
    "Video source",
    ["Sample video library", "Upload your own video"],
    horizontal=True,
)

input_path = None
temporary_input = False

if source_mode == "Sample video library":
    if not sample_videos:
        st.warning("No sample videos found in demo/videos.")
    else:
        selected_video = st.selectbox(
            "Select a sample video",
            sample_videos,
            format_func=lambda p: p.stem.replace("_", " ").title(),
        )
        st.video(str(selected_video))

        if st.button("Analyze selected sample", type="primary"):
            input_path = str(selected_video)

else:
    uploaded = st.file_uploader(
        "Upload a road video",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded is not None:
        st.video(uploaded)

        if st.button("Analyze uploaded video", type="primary"):
            with tempfile.NamedTemporaryFile(
                suffix=Path(uploaded.name).suffix or ".mp4",
                delete=False,
            ) as temp:
                temp.write(uploaded.getbuffer())
                input_path = temp.name
                temporary_input = True

if input_path is not None:
    output_path = None

    try:
        model, device = load_model(str(MODEL_PATH))
        st.info(f"Using device: {device}")

        with st.spinner("Analyzing video with the trained model..."):
            results, output_path, fps, total_frames = analyze_video(
                input_path, model, device
            )

        st.success("Analysis completed.")

        st.subheader("Video Analysis Overview")
        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Total Video Frames",
            f"{total_frames:,}",
        )
        metric2.metric(
            "Frames Analyzed",
            f"{len(results):,}",
            help=(
                "Frames processed by the model using your selected frame step "
                "and frame limit."
            ),
        )

        mean_score = float(results["positive_class_score"].mean())
        metric3.metric(
            "Mean Positive-Class Score",
            f"{mean_score:.3f}",
            help=(
                "Average model score for the analyzed frames. This is not a "
                "calibrated collision probability."
            ),
        )

        st.caption(f"Video FPS: {fps:.2f} | Threshold: {threshold:.2f}")

        st.subheader("Annotated Video")
        video_bytes = Path(output_path).read_bytes()
        st.video(video_bytes)
        st.download_button(
            "Download Annotated Video",
            data=video_bytes,
            file_name="annotated_collision_video.mp4",
            mime="video/mp4",
        )

        st.subheader("Positive-Class Score Over Time")
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(
            results["timestamp_seconds"],
            results["positive_class_score"],
            linewidth=2,
            label="Positive-class score",
        )
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1.8,
            label=f"Selected threshold ({threshold:.2f})",
        )
        ax.set_xlabel("Video Time (seconds)")
        ax.set_ylabel("Positive-Class Score")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


        # --- PROFESSIONAL ANALYSIS SUMMARY ---
        st.divider()
        st.subheader("📊 Analysis Summary")

        required_columns = {"positive_class_score", "timestamp_seconds"}
        if required_columns.issubset(results.columns) and not results.empty:
            scores = results["positive_class_score"].astype(float)
            timestamps = results["timestamp_seconds"].astype(float)

            flagged = scores >= threshold
            flagged_count = int(flagged.sum())
            analyzed_count = len(scores)
            flagged_percent = 100.0 * flagged_count / analyzed_count

            peak_index = scores.idxmax()
            peak_score = float(scores.loc[peak_index])
            peak_timestamp = float(timestamps.loc[peak_index])

            summary_col1, summary_col2, summary_col3 = st.columns(3)
            summary_col1.metric(
                "Frames flagged",
                f"{flagged_count:,} / {analyzed_count:,}",
            )
            summary_col2.metric(
                "Frames flagged (%)",
                f"{flagged_percent:.1f}%",
            )
            summary_col3.metric(
                "Highest positive-class score",
                f"{peak_score:.3f}",
            )

            st.write(f"**Peak score timestamp:** {peak_timestamp:.2f} seconds")

            report = f"""ROAD COLLISION MODEL — VIDEO ANALYSIS REPORT

        Project: AI-Based Road Collision and Near-Miss Prediction Using Computer Vision

        Analysis overview
        -----------------
        Total video frames: {total_frames}
        Analyzed frames: {analyzed_count}
        Video FPS: {fps:.2f}
        Decision threshold: {threshold:.2f}
        Mean positive-class score: {scores.mean():.4f}
        Highest positive-class score: {peak_score:.4f}
        Timestamp of highest score: {peak_timestamp:.2f} seconds
        Frames at or above threshold: {flagged_count}
        Percentage of analyzed frames at or above threshold: {flagged_percent:.2f}%

        Interpretation
        --------------
        A frame is flagged when its positive-class score is greater than
        or equal to the selected threshold.

        These scores are model outputs, not calibrated collision probabilities.
        A flagged frame does not confirm that a collision will occur, and a
        frame below threshold does not guarantee safety.

        Research demonstration only. Do not use this application as a
        real-time driving safety or collision-warning system.
        """

            st.download_button(
                "📄 Download Analysis Summary (TXT)",
                data=report.encode("utf-8"),
                file_name="road_collision_analysis_report.txt",
                mime="text/plain",
            )
        else:
            st.warning(
                "Summary metrics are unavailable because the results table "
                "is empty or is missing required columns."
            )
        st.subheader("Frame Results")
        st.dataframe(results, use_container_width=True)
        st.download_button(
            "Download CSV Results",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="collision_analysis_results.csv",
            mime="text/csv",
        )

    except Exception as error:
        st.error(f"Analysis failed: {error}")

    finally:
        if temporary_input and Path(input_path).exists():
            try:
                Path(input_path).unlink()
            except OSError:
                pass

        if output_path and Path(output_path).exists():
            try:
                Path(output_path).unlink()
            except OSError:
                pass
else:
    st.info("Choose a sample video or upload your own, then start analysis.")

