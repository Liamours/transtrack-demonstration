from pathlib import Path


def test_streamlit_defaults_are_fixed():
    source = Path("src/transtrack_demo/streamlit_app.py").read_text()

    assert 'DEFAULT_MODEL_PATH = "models/classifier/best_val_f1.pth"' in source
    assert 'DEFAULT_LOG_DIR = "logs"' in source
    assert "INFER_EVERY_S" in source
    assert "FEATURE_FPS" in source
    assert 'st.sidebar.selectbox("Backend"' not in source
    assert 'st.sidebar.text_input("Model"' not in source
    assert 'st.sidebar.number_input("Clip seconds"' not in source
    assert 'st.sidebar.number_input("Infer every seconds"' not in source
    assert 'st.sidebar.text_input("CSV log"' not in source
    assert "Show model input box" not in source
    assert "_draw_model_input_box" not in source
    assert "Show landmarks" in source
    assert "Show EAR/MAR" in source
    assert "Show label distribution" not in source
    assert "draw_ear_mar" in source
    assert "draw_stats" in source
    assert "zoom_landmark_region" in source
    assert "st.metric" not in source
    assert "st.progress" not in source
    assert "st.bar_chart" not in source


def test_streamlit_uses_webrtc():
    source = Path("src/transtrack_demo/streamlit_app.py").read_text()

    assert "webrtc_streamer" in source
    assert "VideoProcessorBase" in source
    assert "FatigueProcessor" in source
    assert 'st.sidebar.button("Start"' not in source
    assert 'st.sidebar.button("Stop"' not in source
    assert "cv2.VideoCapture" not in source


def test_streamlit_log_path_is_timestamped():
    source = Path("src/transtrack_demo/streamlit_app.py").read_text()

    assert 'strftime("%Y%m%d_%H%M%S")' in source
    assert "_streamlit_inference.csv" in source
