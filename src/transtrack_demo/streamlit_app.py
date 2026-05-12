import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import streamlit as st

from .alarm import alarm_wav_bytes, autoplay_audio_html
from .camera import camera_name, list_cameras
from .inference_worker import InferenceWorker
from .live_inference import BACKENDS
from .stats import FatigueStats, WARNING_LABELS
from .visual_features import VisualFeatureExtractor, draw_ear_mar, draw_landmarks, draw_stats

DEFAULT_BACKEND = "dshow"
DEFAULT_MODEL_PATH = "models/classifier/best_val_f1.pth"
DEFAULT_CLIP_SECONDS = 20
DEFAULT_INFER_EVERY = 10
DEFAULT_LOG_DIR = "logs"


def _camera_options():
    cameras = list_cameras()
    if not cameras:
        return {0: "Camera 0"}
    return {camera["index"]: camera["name"] for camera in cameras}


def _result_panel(result):
    if result is None:
        st.info("Collecting frames before first inference.")
        return

    warning = result["label"] in WARNING_LABELS
    if warning:
        st.warning(f"WARNING: {result['label']} ({result['confidence']:.4f})")
    else:
        st.success(f"OK: {result['label']} ({result['confidence']:.4f})")


def _video_details(fps, width, height, clip_seconds, infer_every, needed_frames):
    st.subheader("Video Specificity")
    st.write(
        {
            "camera_fps": fps,
            "resolution": f"{width}x{height}",
            "rolling_clip_seconds": clip_seconds,
            "captured_frames_per_clip": needed_frames,
            "inference_interval_seconds": infer_every,
            "model_sampling_fps": 10,
            "model_timesteps": 200,
            "model_window_seconds": 20,
        }
    )


def _new_log_path():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(DEFAULT_LOG_DIR) / f"{timestamp}_streamlit_inference.csv")


def _run_stream(
    camera_index,
    backend,
    model_path,
    clip_seconds,
    infer_every,
    log_path,
    alarm_enabled,
    show_landmarks,
    show_ear_mar,
):
    capture = cv2.VideoCapture(camera_index, BACKENDS[backend])
    if not capture.isOpened():
        st.error(f"Camera {camera_index} could not be opened.")
        return

    fps = int(capture.get(cv2.CAP_PROP_FPS) or 10)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    needed_frames = max(1, fps * clip_seconds)
    infer_interval = max(1, fps * infer_every)

    frames = deque(maxlen=needed_frames)
    stats = FatigueStats()
    frame_box = st.empty()
    result_box = st.empty()
    alarm_box = st.empty()
    details_box = st.empty()
    feature_extractor = None

    worker = InferenceWorker(
        model_path=model_path,
        runtime_dir="runtime",
        log_path=log_path,
        fps=fps,
        size=(width, height),
    )
    worker.start()
    last_seen_result = None
    frame_index = 0
    missed_reads = 0
    alarm_index = 0

    with details_box.container():
        _video_details(fps, width, height, clip_seconds, infer_every, needed_frames)

    try:
        if show_landmarks or show_ear_mar:
            feature_extractor = VisualFeatureExtractor()

        while st.session_state.get("stream_running", False):
            ok, frame = capture.read()
            if not ok:
                missed_reads += 1
                if missed_reads >= 30:
                    st.error("Camera frame could not be read.")
                    break
                time.sleep(0.05)
                continue
            missed_reads = 0

            frame_index += 1
            frames.append(frame.copy())

            if len(frames) == needed_frames and frame_index % infer_interval == 0:
                worker.submit(frames)

            result = worker.latest_result
            if result is not None and result is not last_seen_result:
                stats.update(result)
                last_seen_result = result
                if alarm_enabled and result["label"] in WARNING_LABELS:
                    alarm_index += 1
                    alarm_box.markdown(
                        autoplay_audio_html(alarm_wav_bytes(), key=alarm_index),
                        unsafe_allow_html=True,
                    )

            display_frame = frame.copy()
            features = None
            if feature_extractor is not None:
                features = feature_extractor.analyze(display_frame)
                if show_landmarks:
                    draw_landmarks(display_frame, features["landmarks"])
                if show_ear_mar:
                    draw_ear_mar(display_frame, features)
            draw_stats(display_frame, stats)

            frame_box.image(
                cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )

            with result_box.container():
                _result_panel(result)

            time.sleep(0.01)
    finally:
        if feature_extractor is not None:
            feature_extractor.close()
        worker.stop()
        capture.release()


def main():
    st.set_page_config(page_title="TransTrack Demo", layout="wide")
    st.title("TransTrack Fatigue Detection")

    cameras = _camera_options()
    camera_index = st.sidebar.selectbox(
        "Camera",
        options=list(cameras.keys()),
        format_func=lambda index: f"{index} - {cameras[index]}",
    )
    alarm_enabled = st.sidebar.checkbox("Alarm on fatigue", value=True)
    show_landmarks = st.sidebar.checkbox("Show landmarks", value=False)
    show_ear_mar = st.sidebar.checkbox("Show EAR/MAR", value=False)

    st.sidebar.caption(f"Selected: {camera_name(camera_index)}")

    if not Path(DEFAULT_MODEL_PATH).exists():
        st.error(f"Model not found: {DEFAULT_MODEL_PATH}")
        return

    if "stream_running" not in st.session_state:
        st.session_state.stream_running = False
    if "stream_log_path" not in st.session_state:
        st.session_state.stream_log_path = _new_log_path()

    button_label = "Stop" if st.session_state.stream_running else "Start"
    button_type = "secondary" if st.session_state.stream_running else "primary"
    if st.sidebar.button(button_label, type=button_type):
        st.session_state.stream_running = not st.session_state.stream_running
        if st.session_state.stream_running:
            st.session_state.stream_log_path = _new_log_path()

    if st.session_state.get("stream_running", False):
        st.sidebar.caption(f"Log: {st.session_state.stream_log_path}")
        _run_stream(
            camera_index,
            DEFAULT_BACKEND,
            DEFAULT_MODEL_PATH,
            DEFAULT_CLIP_SECONDS,
            DEFAULT_INFER_EVERY,
            st.session_state.stream_log_path,
            alarm_enabled,
            show_landmarks,
            show_ear_mar,
        )
    else:
        st.info("Press Start to begin camera inference.")


if __name__ == "__main__":
    main()
