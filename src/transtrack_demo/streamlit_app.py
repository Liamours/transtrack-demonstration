import os
import time
import threading
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import asyncio
import streamlit as st
import streamlit.components.v1 as components

from collections import deque
from datetime import datetime
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import av
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, RTCConfiguration

from .alarm import alarm_wav_bytes, alarm_js_html
from .logger import CsvLogger
from .pipeline import (
    make_landmarker, extract_frame,
    _load_model, _prepare,
    CLASS_NAMES, SEQUENCE_LENGTH,
)
from .stats import FatigueStats, WARNING_LABELS
from .visual_features import draw_landmarks, draw_ear_mar, draw_stats, zoom_landmark_region

DEFAULT_MODEL_PATH = "models/classifier/best_val_f1.pth"
DEFAULT_LOG_DIR    = "logs"
INFER_EVERY_S      = 10.0
FEATURE_FPS        = 10.0

_RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Suppress aioice cleanup noise on Python 3.14: STUN retry timers fire after
# the UDP transport's _sock/_loop are already set to None on close.
def _aioice_error_filter(loop, context):
    exc = context.get("exception")
    if isinstance(exc, AttributeError) and "'NoneType'" in str(exc):
        return
    loop.default_exception_handler(context)

try:
    asyncio.get_event_loop().set_exception_handler(_aioice_error_filter)
except RuntimeError:
    pass


def _new_log_path():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(DEFAULT_LOG_DIR) / f"{ts}_streamlit_inference.csv")


def _result_panel(result):
    if result is None:
        st.info("Collecting frames before first inference.")
        return
    if result["label"] in WARNING_LABELS:
        st.warning(f"WARNING: {result['label']} ({result['confidence']:.4f})")
    else:
        st.success(f"OK: {result['label']} ({result['confidence']:.4f})")


def _process_frame(img, lmk, buf, stats, logger, last_feat_t, last_infer_t,
                   last_lm, model_ref, show_landmarks, show_ear_mar, alarm_event):
    """Shared per-frame processing used by both local and WebRTC modes.
    Returns (out_bgr, zoom_bgr, updated_last_feat_t, updated_last_infer_t, updated_last_lm, result_or_None).
    """
    now = time.time()
    result = None

    if now - last_feat_t >= 1.0 / FEATURE_FPS:
        last_feat_t = now
        feats, lm_points = extract_frame(img, lmk)
        buf.append(feats if feats is not None else [np.nan] * 8)
        last_lm = lm_points

    if len(buf) >= SEQUENCE_LENGTH and now - last_infer_t >= INFER_EVERY_S:
        last_infer_t = now
        arr = np.nan_to_num(np.array(list(buf), dtype=np.float32))
        with torch.no_grad():
            probs    = F.softmax(model_ref()(_prepare(arr)), dim=-1)
            conf, cls = torch.max(probs, dim=-1)
        result = {
            "label":      CLASS_NAMES[cls.item()],
            "class_id":   cls.item(),
            "confidence": round(conf.item(), 4),
            "scores":     {n: round(probs[0, i].item(), 4) for i, n in enumerate(CLASS_NAMES)},
        }
        stats.update(result)
        if logger:
            logger.write(result)
        if result["label"] in WARNING_LABELS:
            alarm_event.set()

    out = img.copy()
    if show_landmarks and last_lm:
        draw_landmarks(out, last_lm)
    if show_ear_mar and buf:
        f = list(buf)[-1]
        ear_l, ear_r, mar = f[0], f[1], f[2]
        ear = None if (np.isnan(ear_l) and np.isnan(ear_r)) else float(np.nanmean([ear_l, ear_r]))
        draw_ear_mar(out, {"ear": ear, "mar": None if np.isnan(float(mar)) else float(mar)})
    draw_stats(out, stats)
    zoom = zoom_landmark_region(out, last_lm or [])

    return out, zoom, last_feat_t, last_infer_t, last_lm, result


# ─── Local camera worker (cv2 thread) ────────────────────────────────────────

class _LocalFatigueWorker:
    """Runs cv2.VideoCapture in a daemon thread. Used for local mode."""

    def __init__(self, cam_idx: int, log_path: str):
        self._lmk          = make_landmarker()
        self._buf          = deque(maxlen=SEQUENCE_LENGTH)
        self._model        = None
        self._last_feat_t  = 0.0
        self._last_infer_t = 0.0
        self._last_lm      = None
        self._stats        = FatigueStats()
        Path(DEFAULT_LOG_DIR).mkdir(parents=True, exist_ok=True)
        self._logger       = CsvLogger(log_path)
        self.log_path      = log_path

        self.result        = None
        self.display_frame = None
        self.zoom_frame    = None
        self.alarm_event   = threading.Event()
        self.show_landmarks = False
        self.show_ear_mar   = False

        self._running = True
        self._cap     = cv2.VideoCapture(cam_idx)
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._running = False
        self._cap.release()

    def _get_model(self):
        if self._model is None:
            self._model = _load_model(
                Path(DEFAULT_MODEL_PATH), "MultiScaleTCN", torch.device("cpu")
            )
        return self._model

    def _run(self):
        while self._running:
            ret, img = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            out, zoom, self._last_feat_t, self._last_infer_t, self._last_lm, res = (
                _process_frame(
                    img, self._lmk, self._buf, self._stats, self._logger,
                    self._last_feat_t, self._last_infer_t, self._last_lm,
                    self._get_model, self.show_landmarks, self.show_ear_mar,
                    self.alarm_event,
                )
            )
            if res is not None:
                self.result = res
            self.display_frame = out
            self.zoom_frame    = zoom


# ─── WebRTC processor ─────────────────────────────────────────────────────────

class FatigueProcessor(VideoProcessorBase):
    show_landmarks: bool = False
    show_ear_mar:   bool = False

    def __init__(self):
        self._lmk          = make_landmarker()
        self._buf          = deque(maxlen=SEQUENCE_LENGTH)
        self._model        = None
        self._last_feat_t  = 0.0
        self._last_infer_t = 0.0
        self._last_lm      = None
        self._stats        = FatigueStats()
        self._logger       = None
        self.log_path      = None
        self.result        = None
        self.zoom_frame    = None
        self.alarm_event   = threading.Event()

    def set_log(self, path: str):
        self.log_path = path
        Path(DEFAULT_LOG_DIR).mkdir(parents=True, exist_ok=True)
        self._logger = CsvLogger(path)

    def _get_model(self):
        if self._model is None:
            self._model = _load_model(
                Path(DEFAULT_MODEL_PATH), "MultiScaleTCN", torch.device("cpu")
            )
        return self._model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        out, zoom, self._last_feat_t, self._last_infer_t, self._last_lm, res = (
            _process_frame(
                img, self._lmk, self._buf, self._stats, self._logger,
                self._last_feat_t, self._last_infer_t, self._last_lm,
                self._get_model, self.show_landmarks, self.show_ear_mar,
                self.alarm_event,
            )
        )
        if res is not None:
            self.result = res
        self.zoom_frame = zoom
        return av.VideoFrame.from_ndarray(out, format="bgr24")


# ─── Shared display helpers ───────────────────────────────────────────────────

def _show_shared(proc, alarm_enabled, frame_col, zoom_col):
    """Display zoom, result panel, and alarm for either mode."""
    zf = proc.zoom_frame
    if zf is not None:
        zoom_col.image(
            cv2.cvtColor(zf, cv2.COLOR_BGR2RGB),
            caption="Zoomed landmark view",
            use_container_width=True,
        )

    _result_panel(proc.result)

    if proc.alarm_event.is_set() and alarm_enabled:
        proc.alarm_event.clear()
        st.session_state.alarm_idx += 1
        components.html(alarm_js_html(alarm_wav_bytes()), height=0)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="TransTrack Demo", layout="wide")
    st.title("TransTrack Fatigue Detection")

    mode = st.sidebar.radio("Mode", ["Local", "Cloud"], horizontal=True)

    alarm_enabled  = st.sidebar.checkbox("Alarm on fatigue", value=True)
    show_landmarks = st.sidebar.checkbox("Show landmarks",   value=False)
    show_ear_mar   = st.sidebar.checkbox("Show EAR/MAR",    value=False)

    if not Path(DEFAULT_MODEL_PATH).exists():
        st.error(f"Model not found: {DEFAULT_MODEL_PATH}")
        return

    if "alarm_idx" not in st.session_state:
        st.session_state.alarm_idx = 0

    # ── Local mode ────────────────────────────────────────────────────────────
    if mode == "Local":
        cam_idx = st.sidebar.number_input("Camera index", min_value=0, value=0, step=1)

        col1, col2 = st.sidebar.columns(2)
        start = col1.button("START")
        stop  = col2.button("STOP")

        worker: _LocalFatigueWorker | None = st.session_state.get("local_worker")

        if start and worker is None:
            worker = _LocalFatigueWorker(int(cam_idx), _new_log_path())
            st.session_state.local_worker = worker

        if stop and worker is not None:
            worker.stop()
            del st.session_state["local_worker"]
            st.rerun()

        if worker is None:
            st.info("Press START to begin camera inference.")
            return

        worker.show_landmarks = show_landmarks
        worker.show_ear_mar   = show_ear_mar
        st.sidebar.caption(f"Log: {worker.log_path}")

        frame_col, zoom_col = st.columns([2, 1])

        df = worker.display_frame
        if df is not None:
            frame_col.image(cv2.cvtColor(df, cv2.COLOR_BGR2RGB), use_container_width=True)

        _show_shared(worker, alarm_enabled, frame_col, zoom_col)

        time.sleep(0.1)
        st.rerun()
        return

    # ── Cloud / WebRTC mode ───────────────────────────────────────────────────
    def make_processor():
        p = FatigueProcessor()
        p.set_log(_new_log_path())
        return p

    frame_col, zoom_col = st.columns([2, 1])
    with frame_col:
        ctx = webrtc_streamer(
            key="fatigue",
            video_processor_factory=make_processor,
            rtc_configuration=_RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
        )

    if not ctx.state.playing:
        st.info("Press START to begin camera inference.")
        return

    proc = ctx.video_processor
    if proc is None:
        time.sleep(0.1)
        st.rerun()

    proc.show_landmarks = show_landmarks
    proc.show_ear_mar   = show_ear_mar

    if proc.log_path:
        st.sidebar.caption(f"Log: {proc.log_path}")

    _show_shared(proc, alarm_enabled, frame_col, zoom_col)

    time.sleep(0.2)
    st.rerun()


if __name__ == "__main__":
    main()
