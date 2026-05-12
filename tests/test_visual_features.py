from types import SimpleNamespace

import numpy as np

from src.transtrack_demo.stats import FatigueStats
from src.transtrack_demo.visual_features import (
    EYE_MOUTH_INDICES,
    draw_ear_mar,
    draw_landmarks,
    draw_stats,
    zoom_landmark_region,
)


def test_draw_landmarks_changes_frame_pixels():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    landmarks = [SimpleNamespace(x=0.5, y=0.5)]

    draw_landmarks(frame, landmarks)

    assert frame.sum() > 0
    assert frame[10, 10, 1] > 0
    assert frame[10, 10, 0] == 0
    assert frame[10, 10, 2] == 0


def test_eye_mouth_indices_only_include_target_points():
    assert len(EYE_MOUTH_INDICES) == 16
    assert {13, 14, 61, 291}.issubset(EYE_MOUTH_INDICES)
    assert {33, 263}.issubset(EYE_MOUTH_INDICES)


def test_draw_ear_mar_changes_frame_pixels():
    frame = np.zeros((90, 220, 3), dtype=np.uint8)

    draw_ear_mar(frame, {"ear": 0.2, "mar": 0.5})

    assert frame.sum() > 0
    assert frame[15, 15].sum() < 765


def test_draw_stats_changes_frame_pixels():
    frame = np.zeros((140, 320, 3), dtype=np.uint8)
    stats = FatigueStats()
    stats.update({"label": "normal"})
    stats.update({"label": "yawning"})

    draw_stats(frame, stats)

    assert frame.sum() > 0


def test_zoom_landmark_region_returns_zoomed_crop():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    landmarks = [SimpleNamespace(x=0.45, y=0.45), SimpleNamespace(x=0.55, y=0.55)]

    zoom = zoom_landmark_region(frame, landmarks, scale=2.0, padding=10)

    assert zoom.shape[0] > 0
    assert zoom.shape[1] > 0
    assert zoom.shape[0] <= frame.shape[0] * 2
    assert zoom.shape[1] <= frame.shape[1] * 2
