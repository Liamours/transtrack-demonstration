from pathlib import Path


def test_streamlit_defaults_are_fixed():
    source = Path("src/transtrack_demo/streamlit_app.py").read_text()

    assert 'DEFAULT_BACKEND = "dshow"' in source
    assert 'DEFAULT_MODEL_PATH = "models/classifier/best_val_f1.pth"' in source
    assert "DEFAULT_CLIP_SECONDS = 20" in source
    assert "DEFAULT_INFER_EVERY = 10" in source
    assert 'DEFAULT_LOG_DIR = "logs"' in source
    assert 'st.sidebar.selectbox("Backend"' not in source
    assert 'st.sidebar.text_input("Model"' not in source
    assert 'st.sidebar.number_input("Clip seconds"' not in source
    assert 'st.sidebar.number_input("Infer every seconds"' not in source
    assert 'st.sidebar.text_input("CSV log"' not in source
    assert "st.bar_chart" not in source
    assert "Show model input box" not in source
    assert "_draw_model_input_box" not in source


def test_streamlit_uses_single_start_stop_button():
    source = Path("src/transtrack_demo/streamlit_app.py").read_text()

    assert 'button_label = "Stop" if st.session_state.stream_running else "Start"' in source
    assert "st.sidebar.button(button_label" in source
    assert 'st.sidebar.button("Start"' not in source
    assert 'st.sidebar.button("Stop"' not in source


def test_streamlit_log_path_is_timestamped():
    source = Path("src/transtrack_demo/streamlit_app.py").read_text()

    assert 'strftime("%Y%m%d_%H%M%S")' in source
    assert 'f"{timestamp}_streamlit_inference.csv"' in source
