import os
import time
import threading
import numpy as np
import torch
import torch.nn.functional as F
import cv2
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

# JS injected into sidebar to enumerate cameras and write ?cam_id= to URL.
# Full reload on selection is intentional — forces WebRTC to restart with new device.
_CAMERA_SELECTOR_JS = """
<style>
  select{width:100%;padding:3px 6px;font-size:13px;border:1px solid #ccc;border-radius:4px;}
</style>
<select id="s" onchange="pick()"><option value="">Default Camera</option></select>
<script>
(async()=>{
  try{
    try{
      const s=await navigator.mediaDevices.getUserMedia({video:true,audio:false});
      s.getTracks().forEach(t=>t.stop());
    }catch(e){}
    const cur=new URLSearchParams(window.parent.location.search).get('cam_id')||'';
    const devs=await navigator.mediaDevices.enumerateDevices();
    devs.filter(d=>d.kind==='videoinput').forEach((c,i)=>{
      const o=new Option(c.label||'Camera '+(i+1),c.deviceId);
      if(c.deviceId===cur)o.selected=true;
      document.getElementById('s').add(o);
    });
  }catch(e){}
})();
function pick(){
  const v=document.getElementById('s').value;
  const u=new URL(window.parent.location.href);
  v?u.searchParams.set('cam_id',v):u.searchParams.delete('cam_id');
  window.parent.location.replace(u.toString());
}
</script>
"""


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


class FatigueProcessor(VideoProcessorBase):
    show_landmarks: bool = False
    show_ear_mar:   bool = False

    def __init__(self):
        self._lmk          = make_landmarker()
        self._buf          = deque(maxlen=SEQUENCE_LENGTH)
        self._mar_hist     = []   # ponytail: matches _is_masked logic used during training
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
        now = time.time()

        if now - self._last_feat_t >= 1.0 / FEATURE_FPS:
            self._last_feat_t = now
            feats, lm_points = extract_frame(img, self._lmk, self._mar_hist)
            self._buf.append(feats if feats is not None else [np.nan] * 8)
            self._last_lm = lm_points

        if len(self._buf) >= SEQUENCE_LENGTH and now - self._last_infer_t >= INFER_EVERY_S:
            self._last_infer_t = now
            arr = np.nan_to_num(np.array(list(self._buf), dtype=np.float32))
            with torch.no_grad():
                probs    = F.softmax(self._get_model()(_prepare(arr)), dim=-1)
                conf, cls = torch.max(probs, dim=-1)
            result = {
                "label":      CLASS_NAMES[cls.item()],
                "class_id":   cls.item(),
                "confidence": round(conf.item(), 4),
                "scores":     {n: round(probs[0, i].item(), 4)
                               for i, n in enumerate(CLASS_NAMES)},
            }
            self._stats.update(result)
            if self._logger:
                self._logger.write(result)
            if result["label"] in WARNING_LABELS:
                self.alarm_event.set()
            self.result = result

        out = img.copy()
        if self.show_landmarks and self._last_lm:
            draw_landmarks(out, self._last_lm)
        if self.show_ear_mar and self._buf:
            f = list(self._buf)[-1]
            ear_l, ear_r, mar = f[0], f[1], f[2]
            ear = None if (np.isnan(ear_l) and np.isnan(ear_r)) else float(np.nanmean([ear_l, ear_r]))
            draw_ear_mar(out, {"ear": ear, "mar": None if np.isnan(float(mar)) else float(mar)})
        draw_stats(out, self._stats)
        self.zoom_frame = zoom_landmark_region(out, self._last_lm or [])

        return av.VideoFrame.from_ndarray(out, format="bgr24")


def main():
    st.set_page_config(page_title="TransTrack Demo", layout="wide")
    st.title("TransTrack Fatigue Detection")

    cam_id = st.query_params.get("cam_id", "")

    alarm_enabled  = st.sidebar.checkbox("Alarm on fatigue", value=True)
    show_landmarks = st.sidebar.checkbox("Show landmarks",   value=False)
    show_ear_mar   = st.sidebar.checkbox("Show EAR/MAR",    value=False)

    with st.sidebar:
        st.caption("Camera")
        components.html(_CAMERA_SELECTOR_JS, height=36)

    if not Path(DEFAULT_MODEL_PATH).exists():
        st.error(f"Model not found: {DEFAULT_MODEL_PATH}")
        return

    if "alarm_idx" not in st.session_state:
        st.session_state.alarm_idx = 0

    def make_processor():
        p = FatigueProcessor()
        p.set_log(_new_log_path())
        return p

    frame_col, zoom_col = st.columns([2, 1])
    with frame_col:
        video_constraint = {"deviceId": {"ideal": cam_id}} if cam_id else True
        ctx = webrtc_streamer(
            key=f"fatigue-{cam_id}",
            video_processor_factory=make_processor,
            rtc_configuration=_RTC_CONFIG,
            media_stream_constraints={"video": video_constraint, "audio": False},
        )

    zoom_slot = zoom_col.empty()

    # Show log path on sidebar interactions (full reruns only)
    if ctx.state.playing:
        p = ctx.video_processor
        if p and p.log_path:
            st.sidebar.caption(f"Log: {p.log_path}")

    if not ctx.state.playing:
        st.info("Press START to begin camera inference.")
        return

    # Anti-flicker: fragment reruns every 0.2s without re-rendering the whole page.
    # Captures ctx/show_*/alarm_enabled/zoom_slot from main()'s last full run.
    @st.fragment(run_every=0.2)
    def live_panel():
        proc = ctx.video_processor
        if proc is None:
            return

        proc.show_landmarks = show_landmarks
        proc.show_ear_mar   = show_ear_mar

        zf = proc.zoom_frame
        if zf is not None:
            zoom_slot.image(
                cv2.cvtColor(zf, cv2.COLOR_BGR2RGB),
                caption="Zoomed landmark view",
                use_container_width=True,
            )

        _result_panel(proc.result)

        if proc.alarm_event.is_set() and alarm_enabled:
            proc.alarm_event.clear()
            st.session_state.alarm_idx += 1
            components.html(
                alarm_js_html(alarm_wav_bytes(), st.session_state.alarm_idx),
                height=0,
            )

    live_panel()


if __name__ == "__main__":
    main()
