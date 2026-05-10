from pathlib import Path


def test_streamlit_defaults_are_fixed():
    source = Path("src/transtrack_demo/streamlit_app.py").read_text()

    assert 'DEFAULT_BACKEND = "dshow"' in source
    assert 'DEFAULT_MODEL_PATH = "models/classifier/best_val_f1.pth"' in source
    assert "DEFAULT_CLIP_SECONDS = 20" in source
    assert "DEFAULT_INFER_EVERY = 10" in source
    assert 'DEFAULT_LOG_PATH = "logs/streamlit_inference.csv"' in source
    assert 'st.sidebar.selectbox("Backend"' not in source
    assert 'st.sidebar.text_input("Model"' not in source
    assert 'st.sidebar.number_input("Clip seconds"' not in source
    assert 'st.sidebar.number_input("Infer every seconds"' not in source
    assert 'st.sidebar.text_input("CSV log"' not in source
