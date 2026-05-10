from src.transtrack_demo.alarm import alarm_wav_bytes, autoplay_audio_html
from src.transtrack_demo.stats import FatigueStats


def test_fatigue_stats_tracks_warning_rate():
    stats = FatigueStats()
    stats.update({"label": "normal"})
    stats.update({"label": "eyes_closed"})
    stats.update({"label": "yawning"})

    data = stats.as_dict()

    assert data["total"] == 3
    assert data["warning"] == 2
    assert round(data["fatigue_rate"], 4) == 0.6667
    assert data["counts"]["eyes_closed"] == 1
    assert data["counts"]["yawning"] == 1


def test_alarm_audio_html_contains_wav_data():
    audio = alarm_wav_bytes(duration=0.01)
    html = autoplay_audio_html(audio, key=1)

    assert audio.startswith(b"RIFF")
    assert "audio/wav" in html
    assert "base64" in html
    assert 'data-key="1"' in html
