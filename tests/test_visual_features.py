from types import SimpleNamespace

import numpy as np

from src.transtrack_demo.visual_features import draw_landmarks


def test_draw_landmarks_changes_frame_pixels():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    landmarks = [SimpleNamespace(x=0.5, y=0.5)]

    draw_landmarks(frame, landmarks)

    assert frame.sum() > 0
